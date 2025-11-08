import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import hashlib
import random
from sklearn.model_selection import train_test_split


class LabelMeToYOLO:
    def __init__(
        self,
        images_dir,
        json_dir,
        output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    ):
        self.images_dir = Path(images_dir)
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)

        # Split oranları
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # YOLO klasör yapısı
        self.splits = ["train", "valid", "test"]
        for split in self.splits:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        self.class_names = set()
        self.stats = {
            "total_images": 0,
            "total_jsons": 0,
            "matched": 0,
            "duplicate_images": 0,
            "duplicate_jsons": 0,
            "unmatched_images": 0,
            "unmatched_jsons": 0,
        }

    def get_file_hash(self, filepath):
        """Dosyanın hash'ini hesapla (kopya kontrolü için)"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def normalize_filename(self, filename):
        """Dosya adını normalize et (uzantısız)"""
        # Uzantıyı kaldır
        name = Path(filename).stem
        # (1), (2) gibi ekleri kaldır
        if name.endswith(")") and "(" in name:
            base = name.rsplit("(", 1)[0]
            return base
        return name

    def find_image_files(self):
        """Tüm resim dosyalarını bul ve kopyaları tespit et"""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = {}
        image_hashes = defaultdict(list)

        for ext in image_extensions:
            for img_path in self.images_dir.glob(f"*{ext}"):
                images[img_path.name] = img_path
                img_hash = self.get_file_hash(img_path)
                image_hashes[img_hash].append(img_path.name)

        # Kopyaları tespit et
        for hash_val, file_list in image_hashes.items():
            if len(file_list) > 1:
                self.stats["duplicate_images"] += len(file_list) - 1
                print(f"⚠️  Kopya resimler bulundu: {file_list}")

        self.stats["total_images"] = len(images)
        return images

    def find_json_files(self):
        """Tüm JSON dosyalarını bul ve kopyaları tespit et"""
        jsons = {}
        json_hashes = defaultdict(list)

        for json_path in self.json_dir.glob("*.json"):
            jsons[json_path.name] = json_path
            json_hash = self.get_file_hash(json_path)
            json_hashes[json_hash].append(json_path.name)

        # Kopyaları tespit et
        for hash_val, file_list in json_hashes.items():
            if len(file_list) > 1:
                self.stats["duplicate_jsons"] += len(file_list) - 1
                print(f"⚠️  Kopya JSON'lar bulundu: {file_list}")

        self.stats["total_jsons"] = len(jsons)
        return jsons

    def match_files(self, images, jsons):
        """Resim ve JSON dosyalarını eşleştir"""
        matches = []

        # JSON'daki imagePath'i kontrol ederek eşleştir
        for json_name, json_path in jsons.items():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    image_path = data.get("imagePath", "")

                    # İmage path'teki dosya adını al
                    image_name = Path(image_path).name

                    # Resmi bul
                    if image_name in images:
                        matches.append((images[image_name], json_path, image_name))
                        continue

                    # Alternatif: JSON dosya adına göre eşleştir
                    json_base = self.normalize_filename(json_name)

                    for img_name, img_path in images.items():
                        img_base = self.normalize_filename(img_name)
                        if json_base == img_base:
                            matches.append((img_path, json_path, img_name))
                            break
            except Exception as e:
                print(f"❌ JSON okuma hatası ({json_name}): {e}")

        self.stats["matched"] = len(matches)
        return matches

    def polygon_to_yolo(self, points, img_width, img_height):
        """Polygon koordinatlarını YOLO formatına çevir"""
        # Normalize et ve düzleştir
        normalized = []
        for x, y in points:
            normalized.append(x / img_width)
            normalized.append(y / img_height)
        return normalized

    def get_json_classes(self, json_path):
        """JSON dosyasındaki tüm sınıfları al"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            classes = set()
            for shape in data["shapes"]:
                label = shape["label"].lower().strip()
                classes.add(label)
            return classes
        except:
            return set()

    def stratified_split(self, matches):
        """Sınıf dengesini koruyarak veriyi böl"""
        print(f"\n📊 Sınıf dengeli split yapılıyor...")

        # Her örnek için sınıf bilgisini topla
        sample_classes = []
        for img_path, json_path, img_name in matches:
            classes = self.get_json_classes(json_path)
            # Multi-label için en yaygın sınıfı kullan veya birleştir
            class_label = "_".join(sorted(classes)) if classes else "unknown"
            sample_classes.append(class_label)

        # Sınıf dağılımını göster
        class_counts = defaultdict(int)
        for cls in sample_classes:
            class_counts[cls] += 1

        print(f"\n   Sınıf Dağılımı:")
        for cls, count in sorted(class_counts.items()):
            print(f"   - {cls}: {count} örnek")

        # Stratified split (sklearn kullanarak)
        indices = list(range(len(matches)))

        # İlk olarak train ve temp (val+test) olarak ayır
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=42,
            stratify=sample_classes,
        )

        # Temp'i val ve test olarak ayır
        temp_classes = [sample_classes[i] for i in temp_indices]
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)

        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_size),
            random_state=42,
            stratify=temp_classes,
        )

        # İndekslere göre matches'i ayır
        train_matches = [matches[i] for i in train_indices]
        val_matches = [matches[i] for i in val_indices]
        test_matches = [matches[i] for i in test_indices]

        print(f"\n   ✅ Split Sonuçları:")
        print(
            f"   - Train: {len(train_matches)} örnek ({len(train_matches)/len(matches)*100:.1f}%)"
        )
        print(
            f"   - Valid: {len(val_matches)} örnek ({len(val_matches)/len(matches)*100:.1f}%)"
        )
        print(
            f"   - Test: {len(test_matches)} örnek ({len(test_matches)/len(matches)*100:.1f}%)"
        )

        return {"train": train_matches, "valid": val_matches, "test": test_matches}

    def convert_json_to_yolo(self, json_path, output_path, class_to_id):
        """Tek bir JSON dosyasını YOLO formatına çevir"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_width = data["imageWidth"]
            img_height = data["imageHeight"]

            yolo_annotations = []

            for shape in data["shapes"]:
                label = shape["label"].lower().strip()  # Lowercase + boşlukları temizle
                points = shape["points"]

                # Sınıf ID'si
                if label not in class_to_id:
                    class_to_id[label] = len(class_to_id)
                class_id = class_to_id[label]

                # YOLO formatına çevir
                yolo_coords = self.polygon_to_yolo(points, img_width, img_height)

                # YOLO satırı: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
                line = f"{class_id} " + " ".join(
                    [f"{coord:.6f}" for coord in yolo_coords]
                )
                yolo_annotations.append(line)

            # YOLO dosyasına yaz
            with open(output_path, "w") as f:
                f.write("\n".join(yolo_annotations))

            return True
        except Exception as e:
            print(f"❌ Dönüştürme hatası ({json_path.name}): {e}")
            return False

    def create_yaml(self, class_to_id, filename="data.yaml"):
        """YOLO için data.yaml dosyası oluştur"""
        yaml_content = f"""# YOLO Dataset Configuration
path: {self.output_dir.absolute()}
train: train/images
val: valid/images
test: test/images

# Classes
nc: {len(class_to_id)}
names: {list(class_to_id.keys())}
"""
        yaml_path = self.output_dir / ".." / ".." / "configs" / filename
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"✅ data.yaml oluşturuldu: {yaml_path}")

    def convert(self, filename):
        """Ana dönüştürme işlemi"""
        print("🔍 Dosyalar taranıyor...")

        # Dosyaları bul
        images = self.find_image_files()
        jsons = self.find_json_files()

        print(f"\n📊 İstatistikler:")
        print(f"   Toplam resim: {self.stats['total_images']}")
        print(f"   Toplam JSON: {self.stats['total_jsons']}")
        print(f"   Kopya resim: {self.stats['duplicate_images']}")
        print(f"   Kopya JSON: {self.stats['duplicate_jsons']}")

        # Eşleştir
        print(f"\n🔗 Dosyalar eşleştiriliyor...")
        matches = self.match_files(images, jsons)

        print(f"   ✅ Eşleşen çift: {len(matches)}")
        print(f"   ⚠️  Eşleşmeyen resim: {self.stats['total_images'] - len(matches)}")
        print(f"   ⚠️  Eşleşmeyen JSON: {self.stats['total_jsons'] - len(matches)}")

        if len(matches) == 0:
            print("❌ Hiç eşleşme bulunamadı!")
            return

        # Stratified split yap
        split_matches = self.stratified_split(matches)

        # Dönüştür ve kaydet
        print(f"\n🔄 YOLO formatına dönüştürülüyor...")
        class_to_id = {}

        for split_name, split_data in split_matches.items():
            print(f"\n   📁 {split_name.upper()} seti işleniyor...")
            success_count = 0

            for img_path, json_path, img_name in split_data:
                # Resmi kopyala
                output_img = self.output_dir / split_name / "images" / img_name
                shutil.copy2(img_path, output_img)

                # JSON'u YOLO formatına çevir
                output_label = (
                    self.output_dir
                    / split_name
                    / "labels"
                    / f"{Path(img_name).stem}.txt"
                )
                if self.convert_json_to_yolo(json_path, output_label, class_to_id):
                    success_count += 1

            print(f"      ✅ Başarıyla dönüştürülen: {success_count}/{len(split_data)}")

        # Classes dosyası oluştur
        classes_path = self.output_dir / "classes.txt"
        with open(classes_path, "w", encoding="utf-8") as f:
            for class_name in sorted(class_to_id.keys()):
                f.write(f"{class_name}\n")
        print(f"\n✅ classes.txt oluşturuldu")

        # YAML dosyası oluştur
        self.create_yaml(class_to_id, filename)

        # Final istatistikler
        print(f"\n✨ Dönüştürme tamamlandı!")
        print(f"   📂 Çıktı dizini: {self.output_dir}")
        print(f"   🏷️  Sınıf sayısı: {len(class_to_id)}")
        print(f"   📋 Sınıflar: {list(class_to_id.keys())}")
        print(f"\n   📊 Final Dataset:")
        print(f"   - train/images: {len(split_matches['train'])} resim")
        print(f"   - valid/images: {len(split_matches['valid'])} resim")
        print(f"   - test/images: {len(split_matches['test'])} resim")


# Kullanım
if __name__ == "__main__":
    # Dizinleri belirt
    IMAGES_DIR = "./data/temp_v2/IMAGES"  # Resim klasörünüz
    JSON_DIR = "./data/temp_v2/JSON"  # JSON klasörünüz
    OUTPUT_DIR = "./data/data_v2"  # Çıktı klasörü

    # Dönüştürücüyü başlat (70% train, 15% valid, 15% test)
    converter = LabelMeToYOLO(
        IMAGES_DIR,
        JSON_DIR,
        OUTPUT_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    converter.convert(filename="yolov8-m-test.yaml")
