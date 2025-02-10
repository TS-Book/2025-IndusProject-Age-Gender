# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:57:31 2025

@author: thana
"""
import cv2

for i in range(5):  # ลองเช็ค 5 ตัว (เปลี่ยน range ได้)
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"กล้อง {i} พร้อมใช้งาน ✅")
        cap.release()
    else:
        print(f"กล้อง {i} ใช้งานไม่ได้ ❌")
