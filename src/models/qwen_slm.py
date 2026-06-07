import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import io
from PIL import Image

class QwenMedVQA:
    """
    A wrapper class for the Qwen2-VL Small Language Model (SLM) tailored for 
    Medical Visual Question Answering (Med-VQA). It handles model initialization 
    (with optional 4-bit quantization for memory efficiency), image preprocessing, 
    and text generation while preventing memory leaks during inference.
    """
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", use_4bit=True):
        """
        Initializes the Qwen2-VL model and its corresponding processor.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        quantization_config = None
        if use_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" if use_4bit else self.device,
            torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _preprocess_image(self, image):
        """
        Standardizes the input image format to an RGB PIL Image.
        Handles raw bytes dictionaries (from Hugging Face datasets) and file paths.
        """
        if isinstance(image, dict) and 'bytes' in image:
            return Image.open(io.BytesIO(image['bytes'])).convert("RGB")
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image

    def predict(self, image, question, retrieved_cases=None):
        """
        Generates an answer for a given medical image and question.
        Supports passing retrieved visual RAG cases for prompt augmentation.
        """
        from PIL import Image
        
        target_img = self._preprocess_image(image)
        
        if retrieved_cases:
            # Multi-image Visual RAG prompting (identical to testPathVQA.ipynb structure)
            content_block = []
            
            system_instruction = (
                "<System>\n"
                "You are an expert medical diagnostic assistant. Your task is to answer clinical questions "
                "based on an input pathology image and a set of retrieved similar cases.\n"
                "Strictly follow these rules:\n"
                "1. Analyze the provided [Image] and [Question].\n"
                "2. Evaluate the [Retrieved Evidence] from the knowledge base.\n"
                "Note that retrieved cases might have conflicting answers; use them as reference, "
                "but prioritize the visual features of the current image.\n"
                "3. Provide a concise and definitive answer.\n"
                "</System>\n\n"
                "<User>\n"
                "[Retrieved Evidence (Top-K)]\n"
                "Below are similar past cases from the clinical database:\n"
            )
            content_block.append({"type": "text", "text": system_instruction})
            
            # Helper to resize images to avoid VRAM overload
            def optimize_img(pil_img, max_res=336):
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                pil_img.thumbnail((max_res, max_res), Image.Resampling.LANCZOS)
                return pil_img
                
            messages = []
            for i, case in enumerate(retrieved_cases):
                try:
                    ref_img = self._preprocess_image(case['image'])
                    ref_img = optimize_img(ref_img)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ref_img},
                            {"type": "text", "text": f"Reference Case {i+1}:\nQuestion: {case['question']}"}
                        ]
                    })
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"Answer: {case['answer']}"}
                        ]
                    })
                except Exception:
                    continue
            
            target_img = optimize_img(target_img)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": target_img},
                    {"type": "text", "text": f"Current Question: {question}\nBased on the reference cases above, what is the correct answer for the current case? Answer concisely."}
                ]
            })
        else:
            # Fallback to direct direct VQA prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": target_img},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Prevents memory leak during inference loop   
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=15 if retrieved_cases else 50,
                do_sample=False,        
                num_beams=1,            
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        raw_ans = output_text[0].strip()
        raw_ans = raw_ans.replace("</Assistant>", "").replace("<Assistant>", "").strip()
        return raw_ans