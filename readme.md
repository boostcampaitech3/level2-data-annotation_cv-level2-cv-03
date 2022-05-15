# 🔠 Data Annotation for OCR data

## 👨‍🌾 Team

- Level 2 CV Team 03 - 비뜨코인
- 팀 구성원 : 김대근, 박선혁, 강면구, 정재욱, 한현진

## 🎇 Main Subject

스마트폰 카드 결제, 카메라로 카드 인식, 주차장 내 차량 번호 인식기 등 사람이 쓰거나 이미지 속에 있는 문자를 컴퓨터가 인식할 수 있도록 하는 기술을 OCR(Optical Character Recognition)이라 부르며 생활 속 다양한 편의 기능 제공에 필수적인 기술입니다.

해당 프로젝트에서는 크게 글자 검출(text detection), 글자 인식(text recognition), 정렬(Serialization)의 3가지 단계로 구성된 OCR task에서 글자 검출 task 만을 집중적으로 향상 시키는 것을 목표로 합니다.

## 💻 Development Environment

**개발 언어** : PYTHON (IDE: VSCODE, JUPYTER NOTEBOOK)

**서버**: AI STAGES (GPU: NVIDIA TESLA V100)

**협업 Tool** : git, notion, [wandb](https://wandb.ai/cv-3-bitcoin), google spreadsheet, slack

## 🌿 Project Summary

### **Structure**

![프로젝트 플로우 차트](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F4aba9656-0388-4278-804c-1dea3a3c9b69%2FUntitled.png?table=block&id=f22c354b-23dc-4d7d-94a5-60941a03ceae&spaceId=4707137b-2884-4b58-986a-44731422f061&width=2000&userId=554095b8-b4db-49b4-b08d-97e0f08cd382&cache=v2)

프로젝트 플로우 차트

### Dataset

상기한 프로젝트는 두 개의 데이터를 제공했으며 추가적인 데이터 사용에 제한이 없었음

- 기본 제공 데이터
    1. ICDAR17 데이터 중 한글 데이터 (ICDAR17_Korean)
    2. Boostcamp 3기 camper 들이 upstage의 annotation tool을 이용해 직접 생성한 데이터 (Annotated)
- 추가 사용 데이터
    
    ICDAR17 전체 데이터 ⇒ 전체 데이터 중 프로젝트 목표에 맞춰 한글와 영어만 사용 (ICDAR17_MLT)
    
- UFO : upstage에서 제공한 OCR 데이터 표준 format

```markdown
dataset
├── ICDAR17_Korean ┬─ images
|                  └─ ufo ──┬─ train.json 
|                           ├─ train_v1.json
|                           ├─ valid_v1.json
|                           ├─ train_v2.json
|                           └─ valid_v2.json
|
├── Annotated ─────┬─ images 
|                  └─ ufo ──┬─ annotation.json 
|                           └─ train_v3.json
|
└── ICDAR17_MLT ───── raw ──┬─ ch8_training_gt
                            ├─ ch8_training_images
                            ├─ ch8_validation_gt
                            └─ ch8_validation_images
```

### Metrics

□ Precision과 Recall의 조화 평균인 F1-score

단, 하나의 글자 영역을 분리된 다수의 영역으로 예측하는 것을 지양하고자 one-to-many match의 경우 score 0.8로 penalty를 준 형태

□ BBox의 Ground truth와 Prediction은 형태에 따라 아래 세 가지 경우를 생각해볼 수 있음

1. One-to-one match : 하나의 글자 영역이 하나의 예측 영역과 일치하는 경우
2. Many-to-one match : 여러개의 글자 영역이 하나의 예측 영역과 일치하는 경우
3. One-to-many match : 하나의 글자 영역이 여러개의 예측 영역의 합과 일치하는 경우

## [Wrap Up Report](https://www.notion.so/Wrap-Up-ddf7e31aae474ad79b8a4153157ccbf4)
