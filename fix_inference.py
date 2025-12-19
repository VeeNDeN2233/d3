# Временный скрипт для исправления inference_advanced.py
import re

with open('inference_advanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Заменяем проблемный участок
old_code = """    # Загружаем checkpoint для получения сохраненной конфигурации
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", config)
    
    # Используем параметры из сохраненного config или текущего config
    model_config = saved_config.get("model", config.get("model", {}))
    
    if model_type == "bidir_lstm":
        model = BidirectionalLSTMAutoencoder(
            input_size=model_config.get("input_size", config["model"]["input_size"]),
            sequence_length=config["pose"]["sequence_length"],
            encoder_hidden_sizes=model_config.get("encoder_hidden_sizes", config["model"]["encoder_hidden_sizes"]),
            decoder_hidden_sizes=model_config.get("decoder_hidden_sizes", config["model"]["decoder_hidden_sizes"]),
            latent_size=model_config.get("latent_size", config["model"]["latent_size"]),
            num_attention_heads=4,
            dropout=model_config.get("encoder_dropout", config["model"].get("encoder_dropout", 0.2)),
        ).to(device)"""

new_code = """    # Загружаем checkpoint для получения сохраненной конфигурации
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})
    
    # Приоритет: сохраненный config > текущий config
    if saved_config and "model" in saved_config:
        model_config = saved_config["model"]
        logger.info(f"Используется конфигурация из checkpoint: encoder={model_config.get('encoder_hidden_sizes')}, decoder={model_config.get('decoder_hidden_sizes')}, latent={model_config.get('latent_size')}")
    else:
        model_config = config.get("model", {})
        logger.info(f"Используется текущая конфигурация: encoder={model_config.get('encoder_hidden_sizes')}, decoder={model_config.get('decoder_hidden_sizes')}, latent={model_config.get('latent_size')}")
    
    if model_type == "bidir_lstm":
        # Используем значения напрямую из model_config
        encoder_sizes = model_config.get("encoder_hidden_sizes") or [128, 64, 32]
        decoder_sizes = model_config.get("decoder_hidden_sizes") or [64, 128, 75]
        latent_size_val = model_config.get("latent_size") or 32
        input_size_val = model_config.get("input_size") or 75
        
        logger.info(f"Создание модели: encoder={encoder_sizes}, decoder={decoder_sizes}, latent={latent_size_val}")
        
        model = BidirectionalLSTMAutoencoder(
            input_size=input_size_val,
            sequence_length=config["pose"]["sequence_length"],
            encoder_hidden_sizes=encoder_sizes,
            decoder_hidden_sizes=decoder_sizes,
            latent_size=latent_size_val,
            num_attention_heads=4,
            dropout=model_config.get("encoder_dropout", 0.2),
        ).to(device)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('inference_advanced.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed!")
else:
    print("Pattern not found")

