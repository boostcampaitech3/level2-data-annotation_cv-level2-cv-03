# ๐ ย Data Annotation for OCR data
<br />

## ๐จโ๐พ Team

- Level 2 CV Team 03 - ๋น๋จ์ฝ์ธ
- ํ ๊ตฌ์ฑ์ : ๊น๋๊ทผ, ๋ฐ์ ํ, ๊ฐ๋ฉด๊ตฌ, ์ ์ฌ์ฑ, ํํ์ง
<br />

## ๐ Main Subject

์ค๋งํธํฐ ์นด๋ ๊ฒฐ์ , ์นด๋ฉ๋ผ๋ก ์นด๋ ์ธ์, ์ฃผ์ฐจ์ฅ ๋ด ์ฐจ๋ ๋ฒํธ ์ธ์๊ธฐ ๋ฑ ์ฌ๋์ด ์ฐ๊ฑฐ๋ ์ด๋ฏธ์ง ์์ ์๋ ๋ฌธ์๋ฅผ ์ปดํจํฐ๊ฐ ์ธ์ํ  ์ ์๋๋ก ํ๋ ๊ธฐ์ ์ OCR(Optical Character Recognition)์ด๋ผ ๋ถ๋ฅด๋ฉฐ ์ํ ์ ๋ค์ํ ํธ์ ๊ธฐ๋ฅ ์ ๊ณต์ ํ์์ ์ธ ๊ธฐ์ ์๋๋ค.

ํด๋น ํ๋ก์ ํธ์์๋ ํฌ๊ฒ ๊ธ์ ๊ฒ์ถ(text detection), ๊ธ์ ์ธ์(text recognition), ์ ๋ ฌ(Serialization)์ 3๊ฐ์ง ๋จ๊ณ๋ก ๊ตฌ์ฑ๋ OCR task์์ ๊ธ์ ๊ฒ์ถ task ๋ง์ ์ง์ค์ ์ผ๋ก ํฅ์ ์ํค๋ ๊ฒ์ ๋ชฉํ๋ก ํฉ๋๋ค.
<br />

## ๐ป Development Environment

**๊ฐ๋ฐ ์ธ์ด** : PYTHON (IDE: VSCODE, JUPYTER NOTEBOOK)

**์๋ฒ**: AI STAGES (GPU: NVIDIA TESLA V100)

**ํ์ Tool** : git, notion, [wandb](https://wandb.ai/cv-3-bitcoin), google spreadsheet, slack
<br />

## ๐ฟ Project Summary

### **Structure**

![ํ๋ก์ ํธ ํ๋ก์ฐ ์ฐจํธ](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F4aba9656-0388-4278-804c-1dea3a3c9b69%2FUntitled.png?table=block&id=f22c354b-23dc-4d7d-94a5-60941a03ceae&spaceId=4707137b-2884-4b58-986a-44731422f061&width=2000&userId=554095b8-b4db-49b4-b08d-97e0f08cd382&cache=v2)

ํ๋ก์ ํธ ํ๋ก์ฐ ์ฐจํธ

### Dataset

์๊ธฐํ ํ๋ก์ ํธ๋ ๋ ๊ฐ์ ๋ฐ์ดํฐ๋ฅผ ์ ๊ณตํ์ผ๋ฉฐ ์ถ๊ฐ์ ์ธ ๋ฐ์ดํฐ ์ฌ์ฉ์ ์ ํ์ด ์์์

- ๊ธฐ๋ณธ ์ ๊ณต ๋ฐ์ดํฐ
    1. ICDAR17 ๋ฐ์ดํฐ ์ค ํ๊ธ ๋ฐ์ดํฐ (ICDAR17_Korean)
    2. Boostcamp 3๊ธฐ camper ๋ค์ด upstage์ annotation tool์ ์ด์ฉํด ์ง์  ์์ฑํ ๋ฐ์ดํฐ (Annotated)
- ์ถ๊ฐ ์ฌ์ฉ ๋ฐ์ดํฐ
    
    ICDAR17 ์ ์ฒด ๋ฐ์ดํฐ โ ์ ์ฒด ๋ฐ์ดํฐ ์ค ํ๋ก์ ํธ ๋ชฉํ์ ๋ง์ถฐ ํ๊ธ์ ์์ด๋ง ์ฌ์ฉ (ICDAR17_MLT)
    
- UFO : upstage์์ ์ ๊ณตํ OCR ๋ฐ์ดํฐ ํ์ค format

```markdown
dataset
โโโ ICDAR17_Korean โฌโ images
|                  โโ ufo โโโฌโ train.json 
|                           โโ train_v1.json
|                           โโ valid_v1.json
|                           โโ train_v2.json
|                           โโ valid_v2.json
|
โโโ Annotated โโโโโโฌโ images 
|                  โโ ufo โโโฌโ annotation.json 
|                           โโ train_v3.json
|
โโโ ICDAR17_MLT โโโโโ raw โโโฌโ ch8_training_gt
                            โโ ch8_training_images
                            โโ ch8_validation_gt
                            โโ ch8_validation_images
```

### Metrics

โก Precision๊ณผ Recall์ ์กฐํ ํ๊ท ์ธ F1-score

๋จ, ํ๋์ ๊ธ์ ์์ญ์ ๋ถ๋ฆฌ๋ ๋ค์์ ์์ญ์ผ๋ก ์์ธกํ๋ ๊ฒ์ ์ง์ํ๊ณ ์ one-to-many match์ ๊ฒฝ์ฐ score 0.8๋ก penalty๋ฅผ ์ค ํํ

โก BBox์ Ground truth์ Prediction์ ํํ์ ๋ฐ๋ผ ์๋ ์ธ ๊ฐ์ง ๊ฒฝ์ฐ๋ฅผ ์๊ฐํด๋ณผ ์ ์์

1. One-to-one match : ํ๋์ ๊ธ์ ์์ญ์ด ํ๋์ ์์ธก ์์ญ๊ณผ ์ผ์นํ๋ ๊ฒฝ์ฐ
2. Many-to-one match : ์ฌ๋ฌ๊ฐ์ ๊ธ์ ์์ญ์ด ํ๋์ ์์ธก ์์ญ๊ณผ ์ผ์นํ๋ ๊ฒฝ์ฐ
3. One-to-many match : ํ๋์ ๊ธ์ ์์ญ์ด ์ฌ๋ฌ๊ฐ์ ์์ธก ์์ญ์ ํฉ๊ณผ ์ผ์นํ๋ ๊ฒฝ์ฐ
<br />

## [Wrap Up Report](https://www.notion.so/Wrap-Up-ddf7e31aae474ad79b8a4153157ccbf4)
