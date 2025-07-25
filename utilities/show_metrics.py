import matplotlib.pyplot as plt
import json

def read_json_to_list(file_path):
    try:
        with open(f"{file_path}", 'r', encoding='utf-8') as file:
            data = json.load(file)

            # Если данные не являются списком, помещаем их в список
            if not isinstance(data, list):
                data = [data]
        return data
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден.")
        return []
    except json.JSONDecodeError:
        print(f"Ошибка: Файл {file_path} содержит некорректный JSON.")
        return []


if __name__ == "__main__":
    # base_path = "../metrics/model_base"
    base_path = "../metrics/unetvgg"
    # base_path = "../metrics/unet"

    train_losses = read_json_to_list(f"{base_path}/train/train_losses.json")
    val_losses = read_json_to_list(f"{base_path}/validation/val_losses.json")

    # Потери
    plt.plot( range(len(train_losses)), train_losses, "bo", label="Потери на обучении")
    plt.plot(range(len(val_losses)), val_losses, color="orange", label="Потери на валидации")
    plt.title("Потери на этапах обучения и валидации")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    plt.show()