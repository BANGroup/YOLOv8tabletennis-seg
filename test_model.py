from ultralytics import YOLO
import cv2
import os
import glob

# Загружаем обученную модель
model = YOLO('runs/segment/train9/weights/best.pt')

# Путь к папке с тестовыми изображениями
test_dir = 'dataset/images/test/'

# Получаем все изображения в папке
image_files = glob.glob(os.path.join(test_dir, '*.jpg')) + glob.glob(os.path.join(test_dir, '*.jpeg')) + glob.glob(os.path.join(test_dir, '*.png'))

if image_files:
    print(f"Найдено {len(image_files)} изображений для тестирования:")
    for img in image_files:
        print(f"  {os.path.basename(img)}")
    
    print(f"\nТестируем модель на {len(image_files)} изображениях...")
    
    # Запускаем предсказание на всех изображениях
    results = model.predict(
        source=test_dir,
        save=True,
        save_conf=True,
        save_txt=True,
        project='test_results',
        name='batch_test',
        exist_ok=True
    )
    
    print("Предсказание завершено!")
    print(f"Результаты сохранены в: test_results/batch_test/")
    
    # Анализируем результаты
    print("\n📊 Анализ результатов:")
    total_objects = 0
    class_counts = {}
    
    for i, result in enumerate(results):
        image_name = os.path.basename(image_files[i]) if i < len(image_files) else f"image_{i+1}"
        
        if result.masks is not None and result.boxes is not None:
            num_objects = len(result.masks)
            total_objects += num_objects
            print(f"\n🖼️  {image_name}: {num_objects} объектов")
            
            for j, box in enumerate(result.boxes):
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                
                # Подсчет по классам
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                print(f"    {class_name}: {confidence:.1%}")
        else:
            print(f"\n🖼️  {image_name}: Объекты не найдены")
    
    print(f"\n📈 Общая статистика:")
    print(f"   Всего объектов найдено: {total_objects}")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} раз")
        
else:
    print(f"В папке {test_dir} не найдено изображений!") 