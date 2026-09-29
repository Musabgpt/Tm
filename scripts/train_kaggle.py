"""MusabAI Samsung Phone Agent - Kaggle fine-tuning starter.

Designed for a tiny Arabic intent model. The script discovers Samsung/Android
files under /kaggle/input, augments a command seed set, fine-tunes BERT Mini
Arabic, and exports a resumable checkpoint/model.
"""
from pathlib import Path
import json, random, re, os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)

SEED = 20260929
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "asafaya/bert-mini-arabic"
ROOT = Path("/kaggle/working/musabai_samsung_agent")
CKPT = ROOT / "checkpoints"
EXPORT = ROOT / "export" / "musabai_samsung_arabic_nlu"
for p in (CKPT, EXPORT): p.mkdir(parents=True, exist_ok=True)

APPS = ["واتساب","WhatsApp","يوتيوب","YouTube","الرسائل","الهاتف","جهات الاتصال","الكاميرا","الاستديو","المعرض","الإعدادات","Samsung Notes","Samsung Health","SmartThings","Samsung Internet","الساعة","التقويم","الملفات","Google Maps","خرائط جوجل"]
CONTACTS = ["محمد","أحمد","محمود","علي","مصطفى","أبي","أمي","مروة","سارة","خالد"]
WHEN = ["بعد خمس دقائق","بعد عشر دقائق","بعد نصف ساعة","بعد ساعة","بكرا","اليوم الساعة خمسة"]
TEXTS = ["مرحبا","كيفك","وينك","اتصل فيني","تعال","أنا بالطريق","شكرا","صباح الخير"]

INTENTS = {
"OPEN_APP":["افتح {app}","شغل {app}","بدي افتح {app}","روح على {app}","افتحلي {app}"],
"CLOSE_APP":["سكر {app}","اغلق {app}","طلع من {app}","سكرلي {app}"],
"SEND_MESSAGE":["ابعت ل{contact} {text}","رسل ل{contact} {text}","بعت رسالة ل{contact} {text}","ابعث إلى {contact} {text}"],
"READ_MESSAGES":["اقرأ آخر رسالة","جيب آخر رسالة","شو آخر رسالة وصلتني","اقرأ رسائلي"],
"CALL_CONTACT":["اتصل ب{contact}","دق على {contact}","بدي اتصل ب{contact}","رن على {contact}"],
"SET_REMINDER":["ذكرني {when}","اعمل تذكير {when}","نبهني {when}","ذكرني بعد {when}"],
"OPEN_CAMERA":["افتح الكاميرا","شغل الكاميرا","بدي الكاميرا","روح للكاميرا"],
"TAKE_PHOTO":["صور","التقط صورة","خذ صورة","صورلي"],
"OPEN_GALLERY":["افتح الاستديو","افتح المعرض","ورجيني الصور","افتح الصور"],
"VOLUME_UP":["ارفع الصوت","علي الصوت","زيد الصوت","ارفع مستوى الصوت"],
"VOLUME_DOWN":["وطي الصوت","نزل الصوت","خفض الصوت","وطي مستوى الصوت"],
"MUTE":["كتم الصوت","حط صامت","خلي الهاتف صامت","اكتم الصوت"],
"BRIGHTNESS_UP":["ارفع الإضاءة","علي سطوع الشاشة","زيد الإضاءة"],
"BRIGHTNESS_DOWN":["وطي الإضاءة","نزل السطوع","خفض الإضاءة"],
"OPEN_SETTINGS":["افتح الإعدادات","روح على الإعدادات","شغل الإعدادات"],
"OPEN_WIFI_SETTINGS":["افتح إعدادات الواي فاي","روح على الواي فاي","افتح الواي فاي"],
"OPEN_BLUETOOTH_SETTINGS":["افتح البلوتوث","روح على البلوتوث","إعدادات البلوتوث"],
"ENABLE_WIFI":["شغل الواي فاي","فعل الواي فاي","افتح الواي فاي"],
"DISABLE_WIFI":["طفي الواي فاي","سكر الواي فاي","عطل الواي فاي"],
"ENABLE_BLUETOOTH":["شغل البلوتوث","فعل البلوتوث"],
"DISABLE_BLUETOOTH":["طفي البلوتوث","سكر البلوتوث"],
"OPEN_CLOCK":["افتح الساعة","شغل الساعة"],
"OPEN_CALENDAR":["افتح التقويم","شغل التقويم","افتح الكالندر"],
"OPEN_NOTES":["افتح الملاحظات","شغل Samsung Notes","افتح النوتس"],
"SEARCH_PHONE":["دور على {query}","ابحث عن {query} على الهاتف","فتش عن {query}"],
"LOCK_SCREEN":["اقفل الشاشة","سكر الشاشة","قفل الهاتف"],
"GET_BATTERY":["كم البطارية","شو نسبة البطارية","قديش باقي بطارية"],
"OPEN_RECENT_APPS":["افتح التطبيقات الأخيرة","ورجيني آخر التطبيقات","التطبيقات المفتوحة"],
"OPEN_SAMSUNG_SETTINGS":["افتح إعدادات سامسونج","إعدادات الهاتف","إعدادات الجهاز"],
"UNKNOWN":["شو الأخبار","كيفك","احكي معي","ساعدني"]}

QUERIES=["صورة","فيديو","محمد","ملف pdf","Samsung Notes"]

def fill(t):
    return t.format(app=random.choice(APPS), contact=random.choice(CONTACTS), text=random.choice(TEXTS), when=random.choice(WHEN), query=random.choice(QUERIES))

def augment(s):
    reps=[("أرسل","ابعت"),("إلى","ل"),("الآن","هلق"),("الهاتف","الموبايل"),("هاتف","موبايل"),("افتح لي","افتحلي"),("من فضلك",""),("لو سمحت","")]
    for a,b in random.sample(reps, random.randint(0,3)): s=s.replace(a,b)
    return re.sub(r"\s+"," ",s).strip()

def discover_samsung():
    root=Path("/kaggle/input")
    files=[]
    for ext in ("*.csv","*.json","*.jsonl","*.parquet"):
        files.extend(root.rglob(ext))
    return [p for p in files if any(x in str(p).lower() for x in ("samsung","galaxy","android","smartphone","phone"))]

rows=[]
for intent, templates in INTENTS.items():
    for _ in range(2500): rows.append({"text":augment(fill(random.choice(templates))),"label":intent})
seed=Path(__file__).resolve().parents[1]/"data"/"commands_seed.jsonl"
if seed.exists():
    with seed.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))

df=pd.DataFrame(rows).drop_duplicates("text")
labels=sorted(df.label.unique()); label2id={x:i for i,x in enumerate(labels)}; id2label={i:x for x,i in label2id.items()}
train,tmp=train_test_split(df,test_size=.2,random_state=SEED,stratify=df.label)
val,test=train_test_split(tmp,test_size=.5,random_state=SEED,stratify=tmp.label)

tok=AutoTokenizer.from_pretrained(MODEL_NAME)
model=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=len(labels),id2label=id2label,label2id=label2id)
def encode(batch): return tok(batch["text"],truncation=True,max_length=64)
def make(d):
    x=Dataset.from_pandas(d[["text"]].assign(labels=d.label.map(label2id)),preserve_index=False)
    return x.map(encode,batched=True,remove_columns=["text"])
train_ds,val_ds,test_ds=map(make,(train,val,test))

def metrics(p):
    pred=np.argmax(p.predictions,axis=-1)
    return {"accuracy":accuracy_score(p.label_ids,pred),"macro_f1":f1_score(p.label_ids,pred,average="macro")}

args=TrainingArguments(output_dir=str(CKPT),num_train_epochs=20,per_device_train_batch_size=128 if torch.cuda.is_available() else 32,per_device_eval_batch_size=256 if torch.cuda.is_available() else 64,learning_rate=3e-5,weight_decay=.01,warmup_ratio=.05,eval_strategy="steps",eval_steps=250,save_strategy="steps",save_steps=500,save_total_limit=3,logging_steps=50,load_best_model_at_end=True,metric_for_best_model="macro_f1",fp16=torch.cuda.is_available(),report_to="none")
trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,tokenizer=tok,data_collator=DataCollatorWithPadding(tok),compute_metrics=metrics)
ckpts=sorted(CKPT.glob("checkpoint-*"),key=lambda p:int(re.search(r"(\d+)$",p.name).group(1)))
trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
print("VAL",trainer.evaluate(val_ds)); print("TEST",trainer.evaluate(test_ds,metric_key_prefix="test"))
trainer.save_model(str(EXPORT)); tok.save_pretrained(str(EXPORT))
(EXPORT/"command_schema.json").write_text(json.dumps({"labels":labels,"model":MODEL_NAME,"task":"Arabic Samsung phone intent classification"},ensure_ascii=False,indent=2),encoding="utf-8")
print("EXPORTED",EXPORT)
print("Samsung input candidates:",discover_samsung()[:20])
