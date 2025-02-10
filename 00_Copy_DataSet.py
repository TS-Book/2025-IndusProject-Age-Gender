# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:09:53 2025

@author: thana
"""

import os
import shutil

# 📌 กำหนด Path ต้นทางและปลายทาง
source_folder = r"D:\University\3\3_2\Indus based\AGE_Detection\rawdata\age\face_age"
target_folder = r"D:\University\3\3_2\Indus based\AGE_Detection\Datasets"

# 📌 กำหนดช่วงอายุ → Folder
age_to_class = {
    range(0, 13): "0_Child",
    range(13, 21): "1_Teenager",
    range(21, 45): "2_Adult",
    range(45, 65): "3_MiddleAge",
    range(65, 200): "4_Aged",
}

# 📌 วนลูปทุกโฟลเดอร์ใน Source
for folder in sorted(os.listdir(source_folder)):
    folder_path = os.path.join(source_folder, folder)

    # ข้ามถ้าไม่ใช่โฟลเดอร์
    if not os.path.isdir(folder_path):
        continue

    # แปลงชื่อโฟลเดอร์ (001 → 1, 002 → 2, ..., 110 → 110)
    try:
        age = int(folder)
    except ValueError:
        continue  # ข้ามถ้าไม่ใช่ตัวเลข

    # หาว่าอายุอยู่ในช่วงไหน
    target_class = None
    for age_range, class_name in age_to_class.items():
        if age in age_range:
            target_class = class_name
            break

    if target_class is None:
        continue  # ถ้าอายุไม่อยู่ในช่วงที่กำหนดให้ข้าม

    # 📌 โฟลเดอร์ปลายทาง
    class_path = os.path.join(target_folder, target_class)
    os.makedirs(class_path, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

    # 📌 คัดลอกรูปจากโฟลเดอร์อายุไปยังโฟลเดอร์ Dataset
    for filename in sorted(os.listdir(folder_path)):
        file_ext = filename.split(".")[-1].lower()

        # ข้ามถ้าไม่ใช่ไฟล์รูป
        if file_ext not in ["jpg", "jpeg", "png"]:
            continue

        # คัดลอกไฟล์ไปยัง Dataset
        src_file = os.path.join(folder_path, filename)
        dest_file = os.path.join(class_path, filename)

        shutil.copy2(src_file, dest_file)
        print(f"✅ Copy: {src_file} → {dest_file}")

print("🎯 Done! คัดลอกภาพเสร็จเรียบร้อย ✅")
