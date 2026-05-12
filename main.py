import torch
import os
import re
import random
import json
from datasets import load_dataset
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
)
from peft import PeftModel, get_peft_model, LoraConfig
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# ================= НАСТРОЙКИ =================
LIMIT_GQA = 500
LIMIT_MMB = 500
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 2e-5

BASE_DIR = "."
LORA_DIR = os.path.join(BASE_DIR, "finetuned_lora")
RESULTS_DIR = os.path.join(BASE_DIR, "finetuned_results")
# ==============================================================

torch.backends.cudnn.benchmark = True

MODEL_NAME = "deepvk/llava-gemma-2b-lora"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

def load_model_and_processor():
    """Загружает базовую модель (CPU) и процессор, без LoRA."""
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    image_processor.patch_size = 14
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    processor.patch_size = 14
    return model, processor

def train():
    print("\n===== ЗАПУСК ОБУЧЕНИЯ =====")
    model, processor = load_model_and_processor()
    model.config.use_cache = False

    if os.path.isdir(LORA_DIR) and any(fname.endswith(('.bin', '.safetensors')) for fname in os.listdir(LORA_DIR)):
        print("Продолжаем обучение с существующего LoRA-адаптера...")
        model = PeftModel.from_pretrained(model, LORA_DIR, is_trainable=True)
    else:
        print("Создаём новый LoRA с нуля...")
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)

    model.train()
    model = model.to(device)

    print(f"Загрузка GQA‑ru train (лимит {LIMIT_GQA})...")
    inst_stream = load_dataset(
        "deepvk/GQA-ru", "train_balanced_instructions", split="train", streaming=True)
    inst_stream = inst_stream.shuffle(seed=random.randint(0, 10000), buffer_size=10000)
    inst_list = list(inst_stream.take(LIMIT_GQA))
    needed_img_ids = set(it["imageId"] for it in inst_list)
    img_stream = load_dataset("deepvk/GQA-ru", "train_balanced_images", split="train", streaming=True)
    img_map = {}
    for img_item in img_stream:
        if img_item["id"] in needed_img_ids:
            img_map[img_item["id"]] = img_item["image"]
        if len(img_map) >= len(needed_img_ids):
            break
    gqa_data = [{"question": inst["question"], "answer": inst["answer"], "image": img_map[inst["imageId"]]}
                for inst in inst_list if inst["imageId"] in img_map]
    print(f"GQA‑ru загружено: {len(gqa_data)} примеров")

    print(f"Загрузка MMBench‑ru dev (лимит {LIMIT_MMB})...")
    mmb_dev = load_dataset("deepvk/MMBench-ru", split="dev", streaming=True)
    mmb_list = []
    for i, item in enumerate(mmb_dev):
        if i >= LIMIT_MMB + 100:
            break
        choices = item.get("choices", [])
        choices_str = "\n".join(
            f"{chr(65+j)}. {c}" for j, c in enumerate(choices[:6])
        )
        mmb_list.append({
            "question": item["question"],
            "answer": item["answer"],
            "choices_str": choices_str,
            "image": item["image"]
        })
    train_mmb = mmb_list[:LIMIT_MMB]
    print(f"MMBench‑ru обучение: {len(train_mmb)} примеров")

    all_data = [{"type":"gqa", **item} for item in gqa_data] + \
               [{"type":"mmb", **item} for item in train_mmb]
    random.shuffle(all_data)

    class MixedDataset(IterableDataset):
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            for item in self.data:
                yield item

    tok = processor.tokenizer

    def collate_fn(batch):
        texts, images = [], []
        gqa_instructions = [
            "Ответь одним словом.",
            "Ответь кратко.",
            "Одно слово:",
            "Кратко:"
        ]
        mmb_instructions = [
            "Ответь одной буквой.",
            "Выбери букву.",
            "Только букву:",
            "Буква:",
            "Ответь одной буквой."
        ]
        for item in batch:
            if item["type"] == "gqa":
                instr = random.choice(gqa_instructions)
                text = f"USER: <image>\n{item['question']}\n{instr}\nASSISTANT: {item['answer']}{tok.eos_token}"
                texts.append(text)
            else:
                instr = random.choice(mmb_instructions)
                text = f"USER: <image>\n{item['question']}\n{item['choices_str']}\n{instr}\nASSISTANT: {item['answer']}{tok.eos_token}"
                texts.append(text)
            images.append(item["image"])
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    dataset = MixedDataset(all_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress.set_postfix(loss=loss.item())

        model.save_pretrained(LORA_DIR)
        print(f"Адаптер сохранён после {epoch+1} эпохи в {LORA_DIR}")

def evaluate():
    print("\n===== ЗАПУСК ОЦЕНКИ =====")
    if not os.path.isdir(LORA_DIR) or not any(fname.endswith(('.bin', '.safetensors')) for fname in os.listdir(LORA_DIR)):
        print("Папка с адаптером отсутствует или пуста.")
        return

    model, processor = load_model_and_processor()

    print("Применяем LoRA-адаптер для оценки...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    model = model.to(device)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    def load_gqa_test(limit=100):
        inst_ds = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev")
        inst_ds = inst_ds.shuffle(seed=42).select(range(limit))
        needed_ids = set(inst_ds["imageId"])
        img_ds = load_dataset("deepvk/GQA-ru", "testdev_balanced_images", split="testdev")
        img_ds = img_ds.filter(lambda x: x["id"] in needed_ids)
        img_map = {item["id"]: item["image"] for item in img_ds}
        return [{"question": inst["question"], "answer": inst["answer"], "image": img_map[inst["imageId"]]}
                for inst in inst_ds if inst["imageId"] in img_map]

    def load_mmb_test(limit=100, offset=900):
        mmb_dev = load_dataset("deepvk/MMBench-ru", split="dev", streaming=True)
        res = []
        for i, item in enumerate(mmb_dev):
            if i < offset: continue
            if len(res) >= limit: break
            choices = item.get("choices", [])
            letters = [chr(65+j) for j in range(len(choices[:6]))]
            choices_str = "\n".join([f"{l}. {c}" for l, c in zip(letters, choices)])
            res.append({
                "question": item["question"],
                "answer": item["answer"],
                "choices_str": choices_str,
                "image": item["image"]
            })
        return res

    def compute_accuracy(data, task_name):
        correct = total = 0
        results = []
        for item in tqdm(data, desc=f"Оценка {task_name}"):
            if task_name == "gqa_ru":
                prompt = f"USER: <image>\n{item['question']}\nОтветь одним словом без лишних знаков.\nASSISTANT:"
                inputs = processor(text=prompt, images=item["image"], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        num_beams=1,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                    )
                raw = processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
                match = re.search(r'[а-яА-ЯёЁa-zA-Z0-9]+', raw)
                pred = match.group(0).lower() if match else ""
                true = item["answer"].lower()
                is_correct = pred == true

            else:  # mmbench_ru
                prompt = f"USER: <image>\n{item['question']}\n{item['choices_str']}\nОтветь одной буквой.\nASSISTANT:"
                inputs = processor(text=prompt, images=item["image"], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        num_beams=1,
                    )
                raw = processor.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
                pred_letter = raw[0].upper() if raw else ""
                true_letter = item["answer"].strip()[0].upper() if item["answer"] else ""
                is_correct = pred_letter == true_letter
                pred = pred_letter

            correct += int(is_correct)
            total += 1
            results.append({
                "question": item["question"],
                "prediction": pred,
                "ground_truth": item["answer"],
                "correct": is_correct
            })

        acc = correct / total if total else 0.0
        json_path = os.path.join(RESULTS_DIR, f"{task_name}_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": acc, "correct": correct, "total": total, "results": results},
                f, ensure_ascii=False, indent=2
            )
        return acc

    gqa_acc = compute_accuracy(load_gqa_test(100), "gqa_ru")
    mmb_acc = compute_accuracy(load_mmb_test(100), "mmbench_ru")
    print(f"\n=================== РЕЗУЛЬТАТЫ ===================")
    print(f"GQA‑ru accuracy: {gqa_acc:.4f}")
    print(f"MMBench‑ru accuracy: {mmb_acc:.4f}")
    print(f"Результаты сохранены в {RESULTS_DIR}")

def main():
    print("="*50)
    print("  Проект VK VLM")
    print("="*50)
    print("1 - Оценить модель")
    print("2 - Дообучить модель")
    choice = input("Ваш выбор: ").strip()
    if choice == "1":
        evaluate()
    elif choice == "2":
        train()
    else:
        print("Введите 1 или 2.")

if __name__ == "__main__":
    main()