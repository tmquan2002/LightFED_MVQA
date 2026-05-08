import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
import io
from PIL import Image

class LLaVAMedVQA:
    """
    A wrapper class for the LLaVA-Med model tailored for 
    Medical Visual Question Answering (Med-VQA). It handles model initialization 
    (with optional 4-bit quantization for memory efficiency), image preprocessing, 
    and text generation while preventing memory leaks during inference.
    """
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", use_4bit=True):
        """
        Initializes LLaVA-Med model and its corresponding processor.
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
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" if use_4bit else self.device,
            torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _preprocess_image(self, image):
        """
        Standardizes input image format to an RGB PIL Image.
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
        
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        inputs = self.processor(
            text=prompt,
            images=img_obj,
            return_tensors="pt",
        ).to(self.device)

        # Prevents memory leak during inference loop   
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=50,
                do_sample=False,        
                num_beams=3,            
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        
        # Extract only the assistant's response
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[1].strip()
        else:
            response = generated_text
        
        return response
