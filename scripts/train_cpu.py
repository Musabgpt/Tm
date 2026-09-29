"""CPU-only resumable trainer for MusabAI Samsung Phone Agent.

Designed for GitHub Actions without a GPU. Each run trains for a bounded
number of hours, saves a checkpoint, and can resume on the next run.
"""
from pathlib import Path
import json, os, random, re, time
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer

SEED=20260929
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(int(os.getenv('TM_CPU_THREADS','2')))

MODEL='asafaya/bert-mini-arabic'
ROOT=Path('/tmp/tm-training'); CKPT=ROOT/'checkpoints'; EXPORT=ROOT/'export'
CKPT.mkdir(parents=True,exist_ok=True); EXPORT.mkdir(parents=True,exist_ok=True)
HOURS=float(os.getenv('TM_TRAIN_HOURS','5')); EX=int(os.getenv('TM_EXAMPLES_PER_INTENT','1200'))
DEADLINE=time.time()+HOURS*3600

APPS=['واتساب','WhatsApp','يوتيوب','YouTube','الرسائل','الهاتف','الكاميرا','الاستديو','المعرض','الإعدادات','Samsung Notes','Samsung Health','SmartThings','Samsung Internet','الساعة','التقويم','الملفات','Google Maps','خرائط جوجل']
CONTACTS=['محمد','أحمد','محمود','علي','مصطفى','أبي','أمي','مروة','سارة','خالد']
WHEN=['بعد خمس دقائق','بعد عشر دقائق','بعد نصف ساعة','بعد ساعة','بكرا','اليوم الساعة خمسة']
TEXTS=['مرحبا','كيفك','وينك','اتصل فيني','تعال','أنا بالطريق','شكرا','صباح الخير']
Q=['صورة','فيديو','محمد','ملف pdf','Samsung Notes']
INTENTS={
'OPEN_APP':['افتح {app}','شغل {app}','بدي افتح {app}','روح على {app}','افتحلي {app}'],
'CLOSE_APP':['سكر {app}','اغلق {app}','طلع من {app}','سكرلي {app}'],
'SEND_MESSAGE':['ابعت ل{contact} {text}','رسل ل{contact} {text}','بعت رسالة ل{contact} {text}','ابعث إلى {contact} {text}'],
'READ_MESSAGES':['اقرأ آخر رسالة','جيب آخر رسالة','شو آخر رسالة وصلتني','اقرأ رسائلي'],
'CALL_CONTACT':['اتصل ب{contact}','دق على {contact}','بدي اتصل ب{contact}','رن على {contact}'],
'SET_REMINDER':['ذكرني {when}','اعمل تذكير {when}','نبهني {when}','ذكرني بعد {when}'],
'OPEN_CAMERA':['افتح الكاميرا','شغل الكاميرا','بدي الكاميرا'], 'TAKE_PHOTO':['صور','التقط صورة','خذ صورة','صورلي'],
'OPEN_GALLERY':['افتح الاستديو','افتح المعرض','ورجيني الصور'], 'VOLUME_UP':['ارفع الصوت','علي الصوت','زيد الصوت'],
'VOLUME_DOWN':['وطي الصوت','نزل الصوت','خفض الصوت'], 'MUTE':['كتم الصوت','حط صامت','خلي الهاتف صامت'],
'BRIGHTNESS_UP':['ارفع الإضاءة','علي سطوع الشاشة','زيد الإضاءة'], 'BRIGHTNESS_DOWN':['وطي الإضاءة','نزل السطوع','خفض الإضاءة'],
'OPEN_SETTINGS':['افتح الإعدادات','روح على الإعدادات'], 'OPEN_WIFI_SETTINGS':['افتح إعدادات الواي فاي','روح على الواي فاي'],
'OPEN_BLUETOOTH_SETTINGS':['افتح البلوتوث','روح على البلوتوث'], 'ENABLE_WIFI':['شغل الواي فاي','فعل الواي فاي'],
'DISABLE_WIFI':['طفي الواي فاي','سكر الواي فاي'], 'ENABLE_BLUETOOTH':['شغل البلوتوث','فعل البلوتوث'],
'DISABLE_BLUETOOTH':['طفي البلوتوث','سكر البلوتوث'], 'OPEN_CLOCK':['افتح الساعة','شغل الساعة'],
'OPEN_CALENDAR':['افتح التقويم','شغل التقويم','افتح الكالندر'], 'OPEN_NOTES':['افتح الملاحظات','شغل Samsung Notes','افتح النوتس'],
'SEARCH_PHONE':['دور على {query}','ابحث عن {query} على الهاتف','فتش عن {query}'], 'LOCK_SCREEN':['اقفل الشاشة','سكر الشاشة','قفل الهاتف'],
'GET_BATTERY':['كم البطارية','شو نسبة البطارية','قديش باقي بطارية'], 'OPEN_RECENT_APPS':['افتح التطبيقات الأخيرة','ورجيني آخر التطبيقات'],
'OPEN_SAMSUNG_SETTINGS':['افتح إعدادات سامسونج','إعدادات الهاتف','إعدادات الجهاز'], 'UNKNOWN':['شو الأخبار','كيفك','احكي معي','ساعدني']}

def fill(s): return s.format(app=random.choice(APPS),contact=random.choice(CONTACTS),text=random.choice(TEXTS),when=random.choice(WHEN),query=random.choice(Q))
def aug(s):
    for a,b in random.sample([('أرسل','ابعت'),('إلى','ل'),('الآن','هلق'),('الهاتف','الموبايل'),('افتح لي','افتحلي'),('من فضلك',''),('لو سمحت','')],random.randint(0,3)): s=s.replace(a,b)
    return re.sub(r'\s+',' ',s).strip()

rows=[]
for intent,ts in INTENTS.items():
    for _ in range(EX): rows.append({'text':aug(fill(random.choice(ts))),'label':intent})
seed=Path('data/commands_seed.jsonl')
if seed.exists():
    for line in seed.read_text(encoding='utf-8').splitlines():
        if line.strip(): rows.append(json.loads(line))
import pandas as pd
df=pd.DataFrame(rows).drop_duplicates('text')
labels=sorted(df.label.unique()); l2i={x:i for i,x in enumerate(labels)}; i2l={i:x for x,i in l2i.items()}
tr,tmp=train_test_split(df,test_size=.2,random_state=SEED,stratify=df.label); va,te=train_test_split(tmp,test_size=.5,random_state=SEED,stratify=tmp.label)

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=len(labels),id2label=i2l,label2id=l2i)
def enc(b): return tok(b['text'],truncation=True,max_length=64)
def ds(d): return Dataset.from_pandas(d.assign(labels=d.label.map(l2i))[["text","labels"]],preserve_index=False).map(enc,batched=True,remove_columns=['text'])
trd,vad,ted=map(ds,(tr,va,te))
def metric(p):
    z=np.argmax(p.predictions,axis=-1); return {'accuracy':accuracy_score(p.label_ids,z),'macro_f1':f1_score(p.label_ids,z,average='macro')}

# Small CPU batch and short checkpoint interval. max_steps is intentionally large;
# the callback below stops cleanly at the per-run deadline and leaves a checkpoint.
args=TrainingArguments(output_dir=str(CKPT),overwrite_output_dir=False,num_train_epochs=100,per_device_train_batch_size=16,per_device_eval_batch_size=32,gradient_accumulation_steps=2,learning_rate=3e-5,weight_decay=.01,warmup_ratio=.05,eval_strategy='steps',eval_steps=250,save_strategy='steps',save_steps=250,save_total_limit=2,logging_steps=50,load_best_model_at_end=False,report_to='none')

class DeadlineTrainer(Trainer):
    def training_step(self,*args,**kwargs):
        if time.time()>=DEADLINE: raise KeyboardInterrupt('TM_DEADLINE')
        return super().training_step(*args,**kwargs)

trainer=DeadlineTrainer(model=model,args=args,train_dataset=trd,eval_dataset=vad,tokenizer=tok,data_collator=DataCollatorWithPadding(tok),compute_metrics=metric)
ck=sorted(CKPT.glob('checkpoint-*'),key=lambda p:int(re.search(r'(\d+)$',p.name).group(1)))
try:
    trainer.train(resume_from_checkpoint=str(ck[-1]) if ck else None)
except KeyboardInterrupt:
    print('Reached CPU run deadline; checkpoint should be available.')

metrics={'validation':trainer.evaluate(vad),'test':trainer.evaluate(ted,metric_key_prefix='test'),'examples':len(df),'labels':len(labels),'hours_requested':HOURS}
(ROOT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
trainer.save_model(str(EXPORT)); tok.save_pretrained(str(EXPORT))
(EXPORT/'command_schema.json').write_text(json.dumps({'labels':labels,'model':MODEL,'task':'Arabic Samsung phone intent classification'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metrics,ensure_ascii=False,indent=2,default=float))
