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

    def predict(self, image, question):
        """
        Generates an answer for a given medical image and question.
        """
        img_obj = self._preprocess_image(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_obj},
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()