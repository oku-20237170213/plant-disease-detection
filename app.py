"""
app.py — Bitki Hastalığı Tespit Web Uygulaması
Flask backend: model çıkarımı, Grad-CAM, PDF rapor üretimi
"""

import io
import os
import base64
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

from flask import Flask, render_template, request, jsonify, send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Türkçe font kaydı (Windows Arial) ────────────────────────────────────────
def _register_fonts():
    font_paths = [
        (r"C:\Windows\Fonts\arial.ttf",   "Arial"),
        (r"C:\Windows\Fonts\arialbd.ttf", "Arial-Bold"),
    ]
    for path, name in font_paths:
        try:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont(name, path))
        except Exception:
            pass

_register_fonts()
_FONT      = "Arial"      if "Arial"      in pdfmetrics.getRegisteredFontNames() else "Helvetica"
_FONT_BOLD = "Arial-Bold" if "Arial-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

# ── Ayarlar ──────────────────────────────────────────────────────────────────
CHECKPOINT = r"C:\Users\fsemc\Desktop\Plant Disease Detection Project Final\Code\best_model_finetuned.pth"
IMG_SIZE   = 300
DEVICE     = torch.device("cpu")

# ── 72 Sınıf Bilgileri (aciklama · neden · onlem · tedavi) ──────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "bitki": "Elma", "hastalik": "Elma Karasiğer (Scab)", "siddet": "Orta", "saglikli": False,
        "aciklama": "Venturia inaequalis mantarının neden olduğu hastalık. Yaprak ve meyvelerde zeytin yeşili-siyah kadifemsi lekeler oluşturur. Ağır enfeksiyonda yaprak dökülmesi ve meyve çatlakları görülür.",
        "neden":    "Soğuk ve ıslak ilkbahar koşulları sporların hızla yayılmasına zemin hazırlar. Mantar geçen yıl enfekteli yaprak döküntülerinde kışlar.",
        "onlem":    "Dayanıklı çeşit tercih edin. Dökülen yaprakları toplayıp imha edin. Bahçede iyi hava sirkülasyonu sağlayın.",
        "tedavi":   "Tomurcuk patlamasında captan veya mankozeb fungisiti uygulayın. 7-10 günde bir tekrarlayın. Şiddetli enfeksiyonda triazol grubu ilaç kullanın."},
    "Apple___Black_rot": {
        "bitki": "Elma", "hastalik": "Elma Siyah Çürüklük", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Botryosphaeria obtusa mantarı. Yapraklarda mor haleli kahverengi lekeler; meyveler zamanla siyah mumyaya dönüşür.",
        "neden":    "Yaralı veya stres altındaki dokulardan giren mantar; mumyalaşmış meyveler ve enfekteli dallar başlıca bulaşma kaynağıdır.",
        "onlem":    "Mumyalaşmış meyveleri ve kuru dalları uzaklaştırın. Budama aletlerini dezenfekte edin. Dondan, böcekten kaynaklı yaraları minimize edin.",
        "tedavi":   "Enfekteli dalları sağlam dokuya kadar budayın. Bakır bazlı fungisit uygulayın. Kaptan içeren preparatlar etkilidir."},
    "Apple___Cedar_apple_rust": {
        "bitki": "Elma", "hastalik": "Elma-Sedir Pası", "siddet": "Orta", "saglikli": False,
        "aciklama": "Gymnosporangium juniperi-virginianae mantarı. Yaprak üst yüzeyinde parlak sarı-turuncu lekeler; alt yüzeyde tüp şekilli sporlar oluşur.",
        "neden":    "Mantar iki konakçıya ihtiyaç duyar: elma ve ardıç/sedir. Sporlar rüzgarla elma bahçelerine taşınır.",
        "onlem":    "Yakın çevredeki ardıç ve sedir ağaçlarını kaldırın. Dayanıklı elma çeşidi seçin.",
        "tedavi":   "İlkbaharda yaprak açılımından itibaren mikobutanil veya propikonazol uygulayın. 2 haftada bir tekrarlayın."},
    "Apple___healthy": {
        "bitki": "Elma", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Elma bitkisi sağlıklı görünüyor.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez. Dengeli gübreleme ve sulama programına devam edin."},
    "Apricot Normal": {
        "bitki": "Kayısı", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Kayısı bitkisi sağlıklı görünüyor.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Apricot blight leaf disease": {
        "bitki": "Kayısı", "hastalik": "Kayısı Yanıklığı", "siddet": "Orta", "saglikli": False,
        "aciklama": "Kayısı yapraklarında yanıklık hastalığı. Yapraklarda kahverengi-siyah lekeler ve sararma görülür; şiddetli vakalarda tüm dal kuruyabilir.",
        "neden":    "Mantar ve bakteri kaynaklı enfeksiyonlar; yağışlı ve nemli hava koşulları hastalığın yayılımını hızlandırır.",
        "onlem":    "Bahçede hava sirkülasyonunu artırın. Hasta bitki artıklarını imha edin. Sertifikalı fide kullanın.",
        "tedavi":   "Enfekteli dalları budayın. Bakır bazlı fungisit veya bakır oksiklorür uygulayın."},
    "Apricot shot_hole": {
        "bitki": "Kayısı", "hastalik": "Kayısı Kurşun Deliği", "siddet": "Orta", "saglikli": False,
        "aciklama": "Yapraklarda küçük kahverengi lekeler zamanla dökülür ve kurşun deliği görünümü oluşturur. Meyvelerde de yüzeysel lekeler görülebilir.",
        "neden":    "Wilsonomyces carpophilus mantarı; hasat sonrası yağışlı ve nemli hava sporların yayılımını artırır.",
        "onlem":    "Enfekteli yaprak ve meyveleri imha edin. Hasat sonrası bakır içerikli fungisit uygulayın.",
        "tedavi":   "Mankozeb veya bakır bazlı fungisit uygulayın. Enfekteli yaprakları toplayıp yakın."},
    "Cherry_(including_sour)___Powdery_mildew": {
        "bitki": "Kiraz", "hastalik": "Kiraz Külleme", "siddet": "Orta", "saglikli": False,
        "aciklama": "Podosphaera clandestina mantarı. Yaprak yüzeyinde beyaz pudra kaplama görülür; enfekteli yapraklar kıvrılır ve erken dökülür.",
        "neden":    "Kapalı ve nemli ortamlar, aşırı azot gübrelemesi riski artırır. Serin geceler ve ılık gündüzler hastalığı tetikler.",
        "onlem":    "Hava sirkülasyonunu artıracak budama yapın. Aşırı azot kullanımından kaçının. Dayanıklı çeşit seçin.",
        "tedavi":   "Potasyum bikarbonat veya kükürt bazlı fungisit uygulayın. Neem yağı da etkilidir."},
    "Cherry_(including_sour)___healthy": {
        "bitki": "Kiraz", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Kiraz bitkisi sağlıklı görünüyor.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Citrus Die back": {
        "bitki": "Narenciye", "hastalik": "Narenciye Dal Ölümü", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Narenciye dallarının uçtan geriye doğru ölmesi; etkilenen dallar kahverengiye döner ve kurur.",
        "neden":    "Phytophthora ve diğer mantar enfeksiyonları; kuraklık stresi, kötü drenaj ve toprak sıkışması bitkiyi zayıflatarak hastalığa açık hale getirir.",
        "onlem":    "Drenajı iyileştirin. Sulama programını optimize edin. Bitkileri strese sokmayın.",
        "tedavi":   "Enfekteli dalları sağlam dokuya kadar kesin. Bakır bazlı fungisit uygulayın. Yaraları fungisitli macunla örtün."},
    "Citrus Foliage damaged": {
        "bitki": "Narenciye", "hastalik": "Narenciye Yaprak Hasarı", "siddet": "Düşük", "saglikli": False,
        "aciklama": "Narenciye yapraklarında mekanik veya çevresel hasar. Yapraklarda yırtık, delik veya renk değişimi görülür.",
        "neden":    "Böcek zararı, rüzgar, don, mekanik darbe veya kimyasal yanık.",
        "onlem":    "Koruyucu rüzgarkıranlar oluşturun. Don koruması uygulayın. Böcek takibini düzenli yapın.",
        "tedavi":   "Hasar nedenini tespit edin. Böcek kaynaklıysa uygun insektisit uygulayın."},
    "Citrus Healthy leaf": {
        "bitki": "Narenciye", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Narenciye bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Citrus Powdery mildew": {
        "bitki": "Narenciye", "hastalik": "Narenciye Külleme", "siddet": "Orta", "saglikli": False,
        "aciklama": "Narenciyede külleme hastalığı. Yaprak yüzeyinde beyaz pudra kaplama; genç sürgünler deforme olur.",
        "neden":    "Oidium mantar türleri; serin ve nemli hava koşulları, kapalı bahçeler riski artırır.",
        "onlem":    "Hava sirkülasyonunu artırın. Aşırı azotlu gübrelemeden kaçının. Dayanıklı çeşit seçin.",
        "tedavi":   "Kükürt bazlı fungisit veya triadimenol uygulayın. Neem yağı alternatif olarak kullanılabilir."},
    "Citrus Shot hole": {
        "bitki": "Narenciye", "hastalik": "Narenciye Kurşun Deliği", "siddet": "Orta", "saglikli": False,
        "aciklama": "Narenciye yapraklarında kahverengi lekeler zamanla dökülür ve kurşun deliği görünümü oluşturur.",
        "neden":    "Wilsonomyces carpophilus mantarı; ıslak hava koşulları sporların yayılımını hızlandırır.",
        "onlem":    "Enfekteli yaprakları imha edin. Aşırı sulama yapmayın. Hava sirkülasyonunu artırın.",
        "tedavi":   "Bakır bazlı fungisit uygulayın. Mankozeb de etkilidir."},
    "Citrus Yellow dragon": {
        "bitki": "Narenciye", "hastalik": "Narenciye HLB (Turuncu İltihabı)", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Candidatus Liberibacter'ın neden olduğu yıkıcı hastalık. Yapraklarda asimetrik sararma; meyveler küçük, acı ve şekil bozuk kalır.",
        "neden":    "Diaphorina citri (Asya narenciye psyllidi) böceği tarafından taşınır. Enfekteli fide ve aşı gözü de yayılım kaynağıdır.",
        "onlem":    "Sertifikalı ve sağlıklı fide kullanın. Psyllid böceğini sistematik olarak kontrol edin. Enfekteli ağaçları derhal söküp imha edin.",
        "tedavi":   "Bilinen bir tedavisi yoktur. Psyllid vektörünü insektisit ile kontrol edin. Enfekteli ağaçları karantinaya alın."},
    "Citrus Yellow leaves": {
        "bitki": "Narenciye", "hastalik": "Narenciye Yaprak Sararması", "siddet": "Düşük", "saglikli": False,
        "aciklama": "Narenciye yapraklarında sararma. Demir eksikliğinde damarlar yeşil kalırken arası solar; magnezyum eksikliğinde kenardan sararma başlar.",
        "neden":    "Demir, çinko veya magnezyum eksikliği; yüksek toprak pH'ı besin alımını engeller; aşırı sulama kök hasarına yol açar.",
        "onlem":    "Toprak pH'ını 6.0-7.0 aralığında tutun. Dengeli NPK + mikro element gübrelemesi yapın.",
        "tedavi":   "Yapraktan demir sülfat veya şelat demir uygulayın. Toprak pH'ını kükürt ile düzeltin."},
    "Citrus canker": {
        "bitki": "Narenciye", "hastalik": "Narenciye Kankeri", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Xanthomonas citri bakterisi. Yaprak, meyve ve dallarda kabartılı, yağlı görünümlü kabuksu lekeler oluşturur. Meyve değeri tamamen düşer.",
        "neden":    "Rüzgar, yağmur sıçraması ve böcek zararı ile yayılır. Budama aletleri de taşıyıcı olabilir.",
        "onlem":    "Karantina önlemleri alın. Sertifikalı fide kullanın. Aletleri %10 çamaşır suyu ile dezenfekte edin.",
        "tedavi":   "Bakır bakterisit uygulayın. Enfekteli dal ve yaprakları kesip yakın."},
    "Citrus greening": {
        "bitki": "Narenciye", "hastalik": "Narenciye Yeşillenme (HLB)", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "HLB hastalığının erken belirtisi. Yapraklarda düzensiz sarı-yeşil benek; zamanla ağaç verim kaybeder ve ölür.",
        "neden":    "Candidatus Liberibacter; Diaphorina citri psyllid böceği tarafından taşınır.",
        "onlem":    "Psyllid popülasyonunu sürekli izleyin. Sarı yapışkan tuzak kullanın. Sertifikalı fide alın.",
        "tedavi":   "Psyllid vektörünü insektisit ile ilaçlayın. Enfekteli ağaçları sökün."},
    "Citrus mealybugs": {
        "bitki": "Narenciye", "hastalik": "Narenciye Un Böceği", "siddet": "Orta", "saglikli": False,
        "aciklama": "Planococcus citri. Yaplaklarda beyaz pamuksu kitleler; böceğin salgıladığı bal özü is mantarına zemin hazırlar.",
        "neden":    "Sıcak ve kuru ortamlarda çoğalır. Karıncalar un böceklerini korur ve yayılımını artırır.",
        "onlem":    "Karınca popülasyonunu kontrol edin. Doğal düşmanları (Cryptolaemus) destekleyin. Karantinalı fide kullanın.",
        "tedavi":   "Sistemik insektisit (imidakloprid) uygulayın. Yoğun kolonilere alkollü pamukla müdahale edin."},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "bitki": "Mısır", "hastalik": "Mısır Gri Yaprak Lekesi", "siddet": "Orta", "saglikli": False,
        "aciklama": "Cercospora zeae-maydis mantarı. Damarlarla sınırlı, dikdörtgen şekilli gri-bej lekeler; şiddetli vakalarda yaprak tamamen kurur.",
        "neden":    "Nemli ve ıslak hava koşulları sporların yayılımını hızlandırır. Yoğun mısır ekiminde hastalık daha şiddetli seyreder.",
        "onlem":    "Ekim nöbeti uygulayın. Dayanıklı çeşit seçin. Bitki artıklarını imha edin.",
        "tedavi":   "Strobilurin (azoksistrobin) veya triazol (propikonazol) fungisiti uygulayın."},
    "Corn_(maize)___Common_rust_": {
        "bitki": "Mısır", "hastalik": "Mısır Pası", "siddet": "Orta", "saglikli": False,
        "aciklama": "Puccinia sorghi mantarı. Her iki yaprak yüzeyinde tarçın renkli oval püstüller oluşur; şiddetli enfeksiyonda verim önemli ölçüde düşer.",
        "neden":    "Sporlar rüzgarla uzak mesafelere taşınır. Serin ve nemli hava koşulları hastalığı tetikler.",
        "onlem":    "Dayanıklı çeşit kullanın. Erkenden ekin; olgunluk öncesi hasada çalışın.",
        "tedavi":   "Erken belirtilerde triazol grubu fungisit (propikonazol) uygulayın."},
    "Corn_(maize)___Northern_Leaf_Blight": {
        "bitki": "Mısır", "hastalik": "Mısır Kuzey Yaprak Yanıklığı", "siddet": "Orta", "saglikli": False,
        "aciklama": "Setosphaeria turcica mantarı. Yapraklarda büyük puro şekilli gri-yeşil lekeler; ağır enfeksiyonda verim kaybı %50'yi aşabilir.",
        "neden":    "Serin (18-27°C) ve nemli hava; yoğun bitki örtüsü sporlanmayı artırır. Önceki yıl artıkları birincil bulaşma kaynağıdır.",
        "onlem":    "Ekim nöbeti uygulayın. Dayanıklı çeşit seçin. Bitki artıklarını toprağa gömin veya yakın.",
        "tedavi":   "Azoksistrobin veya propikonazol uygulayın. Püskürtmeyi fırlatma döneminden önce yapın."},
    "Corn_(maize)___healthy": {
        "bitki": "Mısır", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Mısır bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Grape___Black_rot": {
        "bitki": "Üzüm", "hastalik": "Üzüm Siyah Çürüklük", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Guignardia bidwellii mantarı. Yapraklarda kahverengi lekeler; üzümler siyah mumyaya dönüşür ve tüm meyve kaybedilebilir.",
        "neden":    "Ilık ve yağışlı hava koşulları; mumyalaşmış meyveler ve enfekteli asma kısımları kışlayan sporları barındırır.",
        "onlem":    "Mumyalaşmış meyveleri ve enfekteli sürgünleri kaldırın. Havalanmayı artıracak budama yapın.",
        "tedavi":   "Tomurcuk patlamasından itibaren mankozeb veya mikobutanil uygulayın. 10-14 günde bir tekrarlayın."},
    "Grape___Esca_(Black_Measles)": {
        "bitki": "Üzüm", "hastalik": "Üzüm Esca Hastalığı", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Karmaşık mantar hastalığı. Yapraklarda kaplan çizgisi deseni; asmada ani solma ve dalların kuruyarak ölmesi görülür.",
        "neden":    "Phaeomoniella chlamydospora ve Phaeoacremonium mantarları; budama yaraları başlıca giriş noktasıdır.",
        "onlem":    "Budama yaralarını hemen fungisitli macunla örtün. Enfekteli odun parçalarını yakın. Sağlıklı fide kullanın.",
        "tedavi":   "Enfekteli dalları kesin, sağlam dokuya kadar budayın. Yaraları trikoderma içerikli biyolojik preparatla koruyun."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "bitki": "Üzüm", "hastalik": "Üzüm Yaprak Yanıklığı", "siddet": "Orta", "saglikli": False,
        "aciklama": "Pseudocercospora vitis mantarı. Yapraklarda koyu kahverengi, düzensiz lekeler; şiddetli vakalarda erken yaprak dökümü olur.",
        "neden":    "Nemli hava koşulları ve yoğun yaprak örtüsü sporlanmayı artırır.",
        "onlem":    "Budama ile havalanmayı artırın. Enfekteli yaprakları imha edin.",
        "tedavi":   "Bakır fungisit veya mankozeb uygulayın."},
    "Grape___healthy": {
        "bitki": "Üzüm", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Asma bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Lemon Anthracnose": {
        "bitki": "Limon", "hastalik": "Limon Antraknoz", "siddet": "Orta", "saglikli": False,
        "aciklama": "Colletotrichum gloeosporioides mantarı. Yaprak ve meyvelerde koyu, çökük lekeler; meyvelerde çürüme ve erken düşme görülür.",
        "neden":    "Nemli ve sıcak hava koşulları; hasarlı doku enfeksiyona açıktır. Sporlar yağmur sıçramasıyla yayılır.",
        "onlem":    "İyi drenaj sağlayın. Hasat aletlerini dezenfekte edin. Enfekteli meyveleri kaldırın.",
        "tedavi":   "Bakır fungisit veya azoksistrobin uygulayın."},
    "Lemon Bacterial Blight": {
        "bitki": "Limon", "hastalik": "Limon Bakteriyel Yanıklık", "siddet": "Orta", "saglikli": False,
        "aciklama": "Pseudomonas syringae bakterisi. Yapraklarda su emmiş, kahverengiye dönen lekeler; sürgünlerde yanıklık görülür.",
        "neden":    "Soğuk ve yağışlı hava; don sonrası yaralar bakteri girişini kolaylaştırır.",
        "onlem":    "Don koruması uygulayın. Budama aletlerini dezenfekte edin. Aşırı sulamadan kaçının.",
        "tedavi":   "Bakır bakterisit uygulayın. Enfekteli dalları budayın ve yaraları macunlayın."},
    "Lemon Citrus Canker": {
        "bitki": "Limon", "hastalik": "Limon Kankeri", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Xanthomonas citri bakterisi. Yaprak ve meyvelerde kabartılı, yağlı görünümlü kabuksu lekeler; meyve piyasa değerini kaybeder.",
        "neden":    "Rüzgar ve yağmurla yayılır; böcek zararından oluşan yaralar giriş noktasıdır.",
        "onlem":    "Sertifikalı fide kullanın. Aletleri dezenfekte edin. Karantina önlemleri alın.",
        "tedavi":   "Bakır bakterisit uygulayın. Enfekteli materyali kesin ve yakın."},
    "Lemon Curl Virus": {
        "bitki": "Limon", "hastalik": "Limon Kıvrılma Virüsü", "siddet": "Orta", "saglikli": False,
        "aciklama": "Citrus leaf curl virus. Yapraklar yukarı kıvrılır, şekil bozulur; genç sürgünler deforme olur.",
        "neden":    "Beyazsinek ve yaprakbiti vektörüyle taşınır. Enfekteli aşı gözü de bulaşma kaynağıdır.",
        "onlem":    "Vektör böcekleri kontrol edin. Sertifikalı, virüsten ari fide kullanın.",
        "tedavi":   "Virüsün doğrudan tedavisi yoktur. Vektör böcekleri insektisit ile kontrol altına alın. Enfekteli bitkileri sökün."},
    "Lemon Deficiency Leaf": {
        "bitki": "Limon", "hastalik": "Limon Besin Eksikliği", "siddet": "Düşük", "saglikli": False,
        "aciklama": "Limon yapraklarında besin eksikliği belirtisi. Demir eksikliğinde damarlar arası solar; magnezyum eksikliğinde kenardan sararma başlar.",
        "neden":    "Azot, demir veya magnezyum eksikliği; yüksek toprak pH'ı besin alımını engeller.",
        "onlem":    "Toprak pH'ını 6.0-7.0 aralığında tutun. Düzenli toprak analizi yaptırın.",
        "tedavi":   "Yapraktan şelat demir veya magnezyum sülfat uygulayın. Toprak pH'ını kükürt ile düzeltin."},
    "Lemon Dry Leaf": {
        "bitki": "Limon", "hastalik": "Limon Yaprak Kuruması", "siddet": "Düşük", "saglikli": False,
        "aciklama": "Limon yapraklarında kuruma ve kahverengileşme. Su stresi veya kök hasarı belirtisi olabilir.",
        "neden":    "Kuraklık stresi, aşırı sulama kaynaklı kök çürüklüğü veya toprak tuzluluğu.",
        "onlem":    "Sulama düzenini gözden geçirin. Drenajı kontrol edin. Tuzlu su kullanmaktan kaçının.",
        "tedavi":   "Sulama programını yeniden düzenleyin. Kök sağlığını kontrol edin; gerekirse fosfona uygulayın."},
    "Lemon Healthy Leaf": {
        "bitki": "Limon", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Limon bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Lemon Sooty Mould": {
        "bitki": "Limon", "hastalik": "Limon İs Mantarı", "siddet": "Düşük", "saglikli": False,
        "aciklama": "Kapnodium mantar türleri. Yaprak yüzeyinde siyah is tabakası; fotosentezi engeller ve meyve piyasa değerini düşürür.",
        "neden":    "Yaprakbit, beyazsinek ve un böceklerinin salgıladığı bal özü üzerinde gelişir. Böceksiz bitkide is mantarı oluşmaz.",
        "onlem":    "Yaprakbit ve beyazsineği kontrol altında tutun. Karınca popülasyonunu azaltın.",
        "tedavi":   "Böceklere karşı insektisit uygulayın. Is tabakasını ıslak bezle silin; bakır içerikli fungisit destekleyici olarak kullanılabilir."},
    "Lemon Spider Mites": {
        "bitki": "Limon", "hastalik": "Limon Kırmızı Örümcek Akarı", "siddet": "Orta", "saglikli": False,
        "aciklama": "Tetranychus urticae. Yapraklarda gümüşi beneklenme ve ince ağ; ağır enfeksiyonda yapraklar dökülür.",
        "neden":    "Sıcak ve kuru hava koşulları; aşırı azotlu gübreleme; bazı insektisitlerin doğal düşmanları öldürmesi patlamalara yol açar.",
        "onlem":    "Düzenli sulama ile nem dengesi kurun. Yırtıcı akarları (Phytoseiidae) destekleyin. Geniş spektrumlu insektisitlerden kaçının.",
        "tedavi":   "Akarisit (abamektin veya bifenazat) uygulayın. Yırtıcı akar salımı yapın. İlaçları rotasyonlu kullanın."},
    "Peach___Bacterial_spot": {
        "bitki": "Şeftali", "hastalik": "Şeftali Bakteriyel Leke", "siddet": "Orta", "saglikli": False,
        "aciklama": "Xanthomonas arboricola pv. pruni bakterisi. Su emmiş küçük lekeler; yapraklarda delikler; meyvelerde derin çukurlar oluşur.",
        "neden":    "Yağmur ve rüzgarla yayılır. Nemli ve ılık hava koşulları enfeksiyonu hızlandırır.",
        "onlem":    "Dayanıklı çeşit seçin. Aşırı azotlu gübrelemeden kaçının. Budama ile havalanmayı artırın.",
        "tedavi":   "İlkbaharda tomurcuk kabarmasından itibaren bakır bakterisit uygulayın."},
    "Peach___healthy": {
        "bitki": "Şeftali", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Şeftali bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Pepper,_bell___Bacterial_spot": {
        "bitki": "Biber", "hastalik": "Biber Bakteriyel Leke", "siddet": "Orta", "saglikli": False,
        "aciklama": "Xanthomonas campestris bakterisi. Yapraklarda su emmiş, sarı haleli koyu lekeler; meyvelerde kabuğumsu lekeler oluşur.",
        "neden":    "Islak hava, yağmur sıçraması ve kirli sulama suyu ile yayılır. Enfekteli tohum başlıca kaynak.",
        "onlem":    "Sertifikalı, hastalıksız tohum kullanın. Damlama sulama tercih edin. Bitki artıklarını imha edin.",
        "tedavi":   "Bakır bakterisit uygulayın. Enfekteli bitki kısımlarını kesin."},
    "Pepper,_bell___healthy": {
        "bitki": "Biber", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Biber bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Potato___Early_blight": {
        "bitki": "Patates", "hastalik": "Patates Erken Yanıklık", "siddet": "Orta", "saglikli": False,
        "aciklama": "Alternaria solani mantarı. Yaşlı yapraklarda hedef tahtası görünümlü halkalı kahverengi lekeler; şiddetli vakalarda verim önemli ölçüde düşer.",
        "neden":    "Sıcak ve nemli hava; bitki stresi (besin eksikliği, kuraklık) enfeksiyon riskini artırır.",
        "onlem":    "Ekim nöbeti uygulayın. Sağlıklı tohum yumrusu seçin. Aşırı azot gübrelemesinden kaçının.",
        "tedavi":   "Klorotalonil veya azoksistrobin uygulayın. 7-10 günde bir tekrarlayın."},
    "Potato___Late_blight": {
        "bitki": "Patates", "hastalik": "Patates Geç Yanıklık", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Phytophthora infestans. Su emmiş lekeler hızla büyür; yaprak alt yüzeyinde beyaz sporlanma; 1-2 haftada tüm tarla çökülebilir.",
        "neden":    "Serin (10-20°C) ve nemli hava; sisli ve yağmurlu günler sporlanmayı patlama noktasına getirir.",
        "onlem":    "Sertifikalı tohum kullanın. Hava tahminini takip edin; risk günlerinde koruyucu fungisit uygulayın. Ekim nöbeti uygulayın.",
        "tedavi":   "Dimetomorf, mankozeb veya fosetil-Al içeren fungisit uygulayın. Enfekteli bitkileri hemen imha edin."},
    "Potato___healthy": {
        "bitki": "Patates", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Patates bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Soybean Bacterial Pustule": {
        "bitki": "Soya", "hastalik": "Soya Bakteriyel Püstül", "siddet": "Düşük", "saglikli": False,
        "aciklama": "Xanthomonas axonopodis. Alt yaprak yüzeyinde küçük kabartılı püstüller; şiddetli vakalarda yaprak dökümü olabilir.",
        "neden":    "Nemli ve sıcak hava; enfekteli bitki artıkları ve tohum başlıca kaynak.",
        "onlem":    "Ekim nöbeti uygulayın. Dayanıklı çeşit seçin. Bitki artıklarını imha edin.",
        "tedavi":   "Bakır bakterisit uygulayın. Sertifikalı tohum kullanın."},
    "Soybean Frogeye Leaf Spot": {
        "bitki": "Soya", "hastalik": "Soya Kurbağa Gözü Lekesi", "siddet": "Orta", "saglikli": False,
        "aciklama": "Cercospora sojina mantarı. Kırmızımsı-kahve kenarlı gri lekeler; fungisit dirençli suşlar sorun yaratabilir.",
        "neden":    "Sıcak ve nemli hava; sporlar rüzgar ve yağmurla taşınır. Enfekteli tohum kaynaktır.",
        "onlem":    "Dayanıklı çeşit seçin. Ekim nöbeti uygulayın. Sertifikalı tohum kullanın.",
        "tedavi":   "Triazol veya strobilurin fungisiti uygulayın. Dirençli suşlara karşı etken madde rotasyonu yapın."},
    "Soybean Healty": {
        "bitki": "Soya", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Soya bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Soybean Rust": {
        "bitki": "Soya", "hastalik": "Soya Pası", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Phakopsora pachyrhizi mantarı. Alt yaprak yüzeyinde kahverengi-kırmızı püstüller; yüzey alanının %80'ini geçince verim kaybı kritik düzeye ulaşır.",
        "neden":    "Sporlar rüzgarla yüzlerce km taşınır. Sıcak (15-28°C) ve nemli hava enfeksiyon riskini patlama noktasına getirir.",
        "onlem":    "Erken ekim yapın. Erken olgunlaşan çeşit seçin. Tahmin sistemlerini takip edin.",
        "tedavi":   "Triazol fungisiti (tebukonazol veya propikonazol) uygulayın. Püskürtmeyi belirtiler görünür görünmez başlatın."},
    "Soybean Sudden Death Syndrome": {
        "bitki": "Soya", "hastalik": "Soya Ani Ölüm Sendromu", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Fusarium virguliforme mantarı. Damarlar arası kloroz ve nekroz; kökler çürür; yapraklar dökülmeden önce solar.",
        "neden":    "Soğuk ve ıslak toprak erken sezonda enfeksiyonu kolaylaştırır. Yoğun ekim ve kötü drenaj riski artırır.",
        "onlem":    "Drenajı iyileştirin. Geç ekimle soğuk toprağı geçin. Ekim nöbeti uygulayın.",
        "tedavi":   "Fluopiram veya sedaksan içerikli tohum ilaçlaması yapın. Kök bölgesine fungisit uygulaması destekleyicidir."},
    "Soybean Target Leaf Spot": {
        "bitki": "Soya", "hastalik": "Soya Hedef Yaprak Lekesi", "siddet": "Orta", "saglikli": False,
        "aciklama": "Corynespora cassiicola mantarı. Hedef tahtası görünümlü halkalı kahverengi lekeler; ağır enfeksiyonda yaprak dökümü olur.",
        "neden":    "Sıcak (25-30°C) ve nemli hava koşulları. Bitki artıkları birincil enfeksiyon kaynağıdır.",
        "onlem":    "Ekim nöbeti uygulayın. Bitki artıklarını imha edin. Hava sirkülasyonunu artırın.",
        "tedavi":   "Azoksistrobin veya difenokona fungisiti uygulayın."},
    "Soybean Yellow Mosaic": {
        "bitki": "Soya", "hastalik": "Soya Sarı Mozaik Virüsü", "siddet": "Orta", "saglikli": False,
        "aciklama": "Bean yellow mosaic virus (BYMV). Yapraklarda sarı-yeşil mozaik desen; bitki bodur kalır ve verim düşer.",
        "neden":    "Yaprakbiti (Aphis craccivora) vektörüyle kalıcı olmayan şekilde taşınır. Enfekteli bitki artıkları kaynak.",
        "onlem":    "Yaprakbitini insektisit ile kontrol edin. Mineral yağ spreyi vektör kontrolüne yardımcı olur. Toleranslı çeşit seçin.",
        "tedavi":   "Virüsün doğrudan tedavisi yoktur. Vektörü kontrol altına alın ve enfekteli bitkileri sökün."},
    "Squash___Powdery_mildew": {
        "bitki": "Kabak", "hastalik": "Kabak Külleme", "siddet": "Orta", "saglikli": False,
        "aciklama": "Podosphaera xanthii mantarı. Yaprak üst yüzeyinde beyaz pudra lekeleri; ağır enfeksiyonda yaprak erken dökülür, meyve olgunlaşması bozulur.",
        "neden":    "Kuru ve sıcak gündüzler ile serin geceler küllemeyi tetikler. Kapalı ortam ve aşırı azot riski artırır.",
        "onlem":    "Hava sirkülasyonunu artıracak sıklıkta ekim yapın. Aşırı azotlu gübrelemeden kaçının. Dayanıklı çeşit seçin.",
        "tedavi":   "Potasyum bikarbonat, neem yağı veya kükürt bazlı fungisit uygulayın."},
    "Strawberry___Leaf_scorch": {
        "bitki": "Çilek", "hastalik": "Çilek Yaprak Yanığı", "siddet": "Orta", "saglikli": False,
        "aciklama": "Diplocarpon earliana mantarı. Küçük koyu mor lekeler birleşerek büyür; yaprak kahverengileşir ve yanar görünümü alır.",
        "neden":    "Yağmurlu hava ve yüksek nem; sporlar yağmur sıçramasıyla yayılır. Uzun süre ıslak yaprak enfeksiyonu kolaylaştırır.",
        "onlem":    "Damlama sulama kullanın. Yaprak ıslaklığını azaltın. Bitki artıklarını imha edin. Çilekleri her 3-4 yılda bir yenileyin.",
        "tedavi":   "Captan veya mikobutanil uygulayın. Enfekteli yaprakları kaldırın."},
    "Strawberry___healthy": {
        "bitki": "Çilek", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Çilek bitkisi sağlıklı.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Tomato___Bacterial_spot": {
        "bitki": "Domates", "hastalik": "Domates Bakteriyel Leke", "siddet": "Orta", "saglikli": False,
        "aciklama": "Xanthomonas perforans bakterisi. Sarı haleli küçük kahverengi lekeler; meyvelerde kabuğumsu lekeler; verim ve piyasa değeri düşer.",
        "neden":    "Yağmur sıçraması ve rüzgarla yayılır. Enfekteli tohum ve fide başlıca kaynaktır.",
        "onlem":    "Sertifikalı, hastalıksız tohum kullanın. Damlama sulamayı tercih edin. Bitki artıklarını imha edin.",
        "tedavi":   "Bakır bakterisit uygulayın. Ağır enfeksiyonda bakır + mancozeb kombinasyonu etkilidir."},
    "Tomato___Early_blight": {
        "bitki": "Domates", "hastalik": "Domates Erken Yanıklık", "siddet": "Orta", "saglikli": False,
        "aciklama": "Alternaria solani mantarı. Yaşlı yapraklarda hedef tahtası görünümlü lekeler; yukarıya doğru ilerler; şiddetli vakalarda verim kaybı büyük.",
        "neden":    "Sıcak ve nemli hava; stres altındaki bitkiler daha duyarlıdır. Bitki artıkları sporları barındırır.",
        "onlem":    "Ekim nöbeti uygulayın. Alt yaprakları temizleyin. Denge gübrelemesi yapın.",
        "tedavi":   "Klorotalonil veya azoksistrobin uygulayın. 7 günde bir tekrarlayın."},
    "Tomato___Late_blight": {
        "bitki": "Domates", "hastalik": "Domates Geç Yanıklık", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Phytophthora infestans. Yapraklarda su emmiş lekeler; alt yüzeyde beyaz sporlanma; hızlı ilerleme; tüm tarla birkaç günde çökebilir.",
        "neden":    "Serin ve nemli hava (15-20°C); sürekli sis veya yağmur; sporlar hızla yayılır.",
        "onlem":    "Hava tahminini takip edin; risk öncesi koruyucu ilaçlama yapın. Aşırı sulamadan kaçının. Ekim nöbeti uygulayın.",
        "tedavi":   "Klorotalonil, mankozeb veya metalaksil+mankozeb uygulayın. Enfekteli bitkileri hemen imha edin."},
    "Tomato___Leaf_Mold": {
        "bitki": "Domates", "hastalik": "Domates Yaprak Küfü", "siddet": "Orta", "saglikli": False,
        "aciklama": "Passalora fulva mantarı. Üst yüzeyde sarı lekeler; alt yüzeyde zeytin yeşili-kahve sporlanma; seralarda sık görülür.",
        "neden":    "Yüksek bağıl nem (>85%) ve yetersiz havalandırma sera ortamında hastalığı tetikler.",
        "onlem":    "Sera havalandırmasını artırın. Sulama sıklığını azaltın. Yaprak ıslaklığını önleyin.",
        "tedavi":   "Mankozeb veya klorotalonil uygulayın. Sera içinde hava sirkülasyon fanı kullanın."},
    "Tomato___Septoria_leaf_spot": {
        "bitki": "Domates", "hastalik": "Domates Septoria Lekesi", "siddet": "Orta", "saglikli": False,
        "aciklama": "Septoria lycopersici mantarı. Beyaz merkezli, koyu kenarlı küçük yuvarlak lekeler; alt yapraklardan başlar yukarı çıkar.",
        "neden":    "Sıcak ve yağışlı hava; sporlar yağmur sıçramasıyla 1 m'ye kadar taşınır.",
        "onlem":    "Alt yaprakları temizleyin. Ekim nöbeti uygulayın. Mulç kullanın.",
        "tedavi":   "Klorotalonil veya mankozeb uygulayın. 7-10 günde bir tekrarlayın."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "bitki": "Domates", "hastalik": "Domates Kırmızı Örümcek", "siddet": "Orta", "saglikli": False,
        "aciklama": "Tetranychus urticae. Yaprak üst yüzeyinde gümüşi beneklenme; alt yüzeyde ince ağ; ağır enfeksiyonda yapraklar solar ve dökülür.",
        "neden":    "Sıcak ve kuru koşullar; bazı insektisitler doğal düşmanları öldürerek patlama yaratır.",
        "onlem":    "Yırtıcı akar (Phytoseiidae) salımı yapın. Geniş spektrumlu insektisitlerden kaçının. Düzenli sulama yapın.",
        "tedavi":   "Abamektin veya bifenazat içerikli akarisit uygulayın. İlaçları rotasyonlu kullanın."},
    "Tomato___Target_Spot": {
        "bitki": "Domates", "hastalik": "Domates Hedef Lekesi", "siddet": "Orta", "saglikli": False,
        "aciklama": "Corynespora cassiicola mantarı. Sarı kenarlı, halkalı kahverengi lekeler önce alt yapraklarda görülür; şiddetli vakalarda yaprak dökümü ve meyve lekesi oluşur.",
        "neden":    "Sıcak (25-30°C) ve nemli hava koşulları; bitki artıkları başlıca spor kaynağıdır. Yoğun gölge ve yetersiz havalandırma hastalığı hızlandırır.",
        "onlem":    "Ekim nöbeti uygulayın. Alt yaprakları erken kaldırın. Hava sirkülasyonunu artıracak budama yapın. Bitki artıklarını imha edin.",
        "tedavi":   "Azoksistrobin veya difenokona içerikli fungisit uygulayın. 7-10 günde bir tekrarlayın."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "bitki": "Domates", "hastalik": "Domates Sarı Yaprak Kıvırcıklık Virüsü", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Bemisia tabaci (beyazsinek) tarafından taşınan virüs. Yapraklar yukarı kıvrılır, sararır; bitki bodurlaşır ve verim önemli ölçüde düşer.",
        "neden":    "Bemisia tabaci beyazsinekleri virüsü bitkiden bitkiye kalıcı olarak taşır. Enfekteli fide ve aşı materyali de yayılım kaynağıdır.",
        "onlem":    "Sertifikalı, virüsten ari fide kullanın. Sarı yapışkan tuzak ile beyazsinek takibini sürdürün. Dayanıklı çeşit seçin.",
        "tedavi":   "Virüsün doğrudan tedavisi yoktur. Beyazsinek vektörünü sistemik insektisit (imidakloprid) ile kontrol altına alın. Enfekteli bitkileri derhal söküp imha edin."},
    "Tomato___Tomato_mosaic_virus": {
        "bitki": "Domates", "hastalik": "Domates Mozaik Virüsü", "siddet": "Orta", "saglikli": False,
        "aciklama": "Tomato mosaic virus (ToMV). Yapraklarda sarı-yeşil mozaik desen, fırfırlı kenarlar; meyveler renk düzensizliği ve lekelenme gösterir.",
        "neden":    "Mekanik temasla (budama, aşılama aletleri, eller) kolayca yayılır. Tütün içenler virüsü doğrudan taşıyabilir. Enfekteli tohum da kaynak olabilir.",
        "onlem":    "Budama aletlerini %10 çamaşır suyu ile dezenfekte edin. Çalışmadan önce sabunla el yıkayın. Sertifikalı tohum kullanın.",
        "tedavi":   "Bilinen bir tedavisi yoktur. Enfekteli bitkileri sökün ve yakın. Aletleri sık sterilize edin."},
    "Tomato___healthy": {
        "bitki": "Domates", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Domates bitkisi sağlıklı görünüyor.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez. Dengeli gübreleme ve sulama programına devam edin."},
    "Watermelon_Downy_Mildew": {
        "bitki": "Karpuz", "hastalik": "Karpuz Mildiyö", "siddet": "Orta", "saglikli": False,
        "aciklama": "Pseudoperonospora cubensis. Yaprak üst yüzeyinde sarı-yeşil lekeler; alt yüzeyde gri-mor sporlanma; ağır enfeksiyonda yapraklar yanar görünümü alır.",
        "neden":    "Serin (15-20°C) ve nemli hava; yoğun çiy ve sis sporları aktive eder. Sporlar rüzgarla uzun mesafelere taşınır.",
        "onlem":    "Hava sirkülasyonunu artıracak sıklıkta ekim yapın. Damlama sulama tercih edin. Yaprak ıslaklığını azaltın.",
        "tedavi":   "Mankozeb veya dimetomorf+mankozeb kombinasyonu uygulayın. Belirtiler görünür görünmez ilaçlamaya başlayın."},
    "Watermelon_Healthy": {
        "bitki": "Karpuz", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Karpuz bitkisi sağlıklı görünüyor.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "Watermelon_Mosaic_Virus": {
        "bitki": "Karpuz", "hastalik": "Karpuz Mozaik Virüsü", "siddet": "Orta", "saglikli": False,
        "aciklama": "Watermelon mosaic virus (WMV). Yapraklarda sarı-yeşil mozaik desen; bitkiler bodurlaşır; meyveler şekil bozukluğu gösterir.",
        "neden":    "Yaprakbiti vektörüyle kalıcı olmayan şekilde yayılır. Enfekteli yabani kabakgiller virüsü barındırabilir.",
        "onlem":    "Yaprakbitini insektisit ve mineral yağ spreyi ile kontrol edin. Virüs rezervuar bitkilerini (yabani kabakgiller) temizleyin.",
        "tedavi":   "Virüsün doğrudan tedavisi yoktur. Vektör yaprakbitini kontrol altına alın. Enfekteli bitkileri söküp imha edin."},
    "cotton_bacterial_blight": {
        "bitki": "Pamuk", "hastalik": "Pamuk Bakteriyel Yanıklık", "siddet": "Orta", "saglikli": False,
        "aciklama": "Xanthomonas axonopodis pv. malvacearum bakterisi. Yapraklarda köşeli, su emmiş lekeler; şiddetli vakalarda dal yanıklığı ve koza hasarı görülür.",
        "neden":    "Yağmur sıçraması ve böcek zararlarıyla yayılır. Enfekteli tohum birincil bulaşma kaynağıdır. Nem ve sıcaklık enfeksiyonu hızlandırır.",
        "onlem":    "Sertifikalı hastalıksız tohum kullanın. Ekim nöbeti uygulayın. Bitki artıklarını imha edin.",
        "tedavi":   "Bakır bakterisit uygulayın. Enfekteli dal ve yaprakları kesin. Tohumluk için asit-delinting uygulaması yapın."},
    "cotton_curl_virus": {
        "bitki": "Pamuk", "hastalik": "Pamuk Yaprak Kıvrılma Virüsü", "siddet": "Yüksek", "saglikli": False,
        "aciklama": "Cotton leaf curl virus (CLCuV). Yapraklar yukarı veya aşağı kıvrılır; damarlarda kalınlaşma; bitkiler bodurlaşır ve verim sıfıra yaklaşır.",
        "neden":    "Bemisia tabaci beyazsinekleri tarafından kalıcı şekilde taşınır. Enfekteli fide ve yabani pamuk türleri rezervuar görevi görür.",
        "onlem":    "Dayanıklı çeşit seçin. Sarı yapışkan tuzak ile beyazsinek izlemesi yapın. Çevre yabani konakçı bitkileri temizleyin.",
        "tedavi":   "Virüsün doğrudan tedavisi yoktur. Beyazsinek vektörünü sistemik insektisit ile kontrol altına alın. Ağır enfekteli bitkileri söküp imha edin."},
    "cotton_healthy": {
        "bitki": "Pamuk", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Pamuk bitkisi sağlıklı görünüyor.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "olive_Healthy": {
        "bitki": "Zeytin", "hastalik": "Sağlıklı", "siddet": "—", "saglikli": True,
        "aciklama": "Hastalık tespit edilmedi. Zeytin bitkisi sağlıklı görünüyor.",
        "neden":    "—", "onlem": "Düzenli gözlem ve standart bakım uygulamalarını sürdürün.",
        "tedavi":   "Herhangi bir ilaçlama gerekmez."},
    "olive_aculus_olearius": {
        "bitki": "Zeytin", "hastalik": "Zeytin Akarı", "siddet": "Orta", "saglikli": False,
        "aciklama": "Aculus olearius (zeytin akarı). Yapraklarda gümüşi-sarımsı renk değişimi; ağır enfeksiyonda yapraklar küçülür ve erken dökülür; verim düşer.",
        "neden":    "Sıcak ve kuru yaz ayları populasyonun patlamasına neden olur. Geniş spektrumlu insektisit kullanımı doğal düşmanları yok ederek riski artırır.",
        "onlem":    "Doğal düşmanları (yırtıcı akarlar, predatör böcekler) koruyun. Geniş spektrumlu insektisit kullanımını sınırlayın. Ağaçları düzenli gözlemleyin.",
        "tedavi":   "Kükürt bazlı akarisit uygulayın. Ağır enfeksiyonda abamektin veya hexythiazox kullanın. Yaz aylarında yağ bazlı preparatlar etkilidir."},
    "olive_peacock_spot": {
        "bitki": "Zeytin", "hastalik": "Zeytin Tavus Kuşu Lekesi", "siddet": "Orta", "saglikli": False,
        "aciklama": "Fusicladium oleagineum mantarı. Yapraklarda sarı hale ile çevrili koyu kahverengi, tavus kuşu gözü şeklinde lekeler; erken yaprak dökümü; verim kaybı.",
        "neden":    "Sonbahar ve kış yağışları sporların yayılımını tetikler. Yoğun yaprak örtüsü ve yetersiz havalandırma hastalığı şiddetlendirir.",
        "onlem":    "Havalanmayı artıracak budama yapın. Dökülen yaprakları bahçeden uzaklaştırın. Aşırı sulama yapmayın.",
        "tedavi":   "Sonbahar ve kışın bakır hidroksit veya bakır oksiklorür fungisiti uygulayın. İki uygulama arasında 45-60 gün bırakın."},
}

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, mdl):
        self.model       = mdl
        self.activations = None
        self.gradients   = None
        # conv_head: blocks'tan sonraki 1x1 conv — temiz gradyan, artık bağlantısı yok.
        # blocks[-1] yerine conv_head kullanmak sağ-üst köşe artefaktını ortadan kaldırır.
        target = mdl.conv_head
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):
        self.activations = out.detach()

    def _bwd(self, _, __, go):
        if go[0] is not None:
            self.gradients = go[0].detach()

    def generate(self, tensor, class_idx):
        self.activations = None
        self.gradients   = None
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            return np.zeros((10, 10), dtype=np.float32)

        # Kanal ağırlıkları
        w   = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1))
        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 0:
            cam = np.array([[float(cam)]])

        # %95 percentile kırpma — tek parlak köşe pikseli tüm haritayı bastırmasın
        vmax = np.percentile(cam, 95)
        vmin = cam.min()
        cam  = np.clip((cam - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return cam

# ── Global değişkenler ────────────────────────────────────────────────────────
_model   = None
_classes = None
_gradcam = None

def load_model():
    global _model, _classes, _gradcam
    ckpt       = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    _classes   = ckpt["classes"]
    model_name = ckpt.get("model_name", "efficientnet_b4")
    n          = len(_classes)

    m = timm.create_model(model_name, pretrained=False, drop_rate=0.0)
    m.classifier = nn.Sequential(nn.Dropout(0.0),
                                  nn.Linear(m.classifier.in_features, n))
    m.load_state_dict(ckpt["model"])
    m = m.to(DEVICE).eval()

    _model   = m
    _gradcam = GradCAM(m)
    print(f"✓ Model: {model_name} | {n} sınıf | {IMG_SIZE}px")

# ── Yardımcılar ───────────────────────────────────────────────────────────────
_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def to_b64(pil_img, size=(IMG_SIZE, IMG_SIZE)):
    buf = io.BytesIO()
    pil_img.resize(size, Image.LANCZOS).save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

def gradcam_pure(cam):
    """Saf ısı haritası (orijinal görüntü olmadan)."""
    cam_r = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    buf   = io.BytesIO()
    Image.fromarray(heat).save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

def gradcam_overlay(pil_img, cam):
    """Grad-CAM + orijinal görüntü bindirmesi."""
    img   = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    cam_r = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heat  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    over  = (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)
    buf   = io.BytesIO()
    Image.fromarray(over).save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # JSON body (base64) veya form-data (file) her ikisini kabul et
    try:
        if request.content_type and "application/json" in request.content_type:
            data_j = request.get_json(force=True)
            img_b64 = data_j.get("image", "")
            img_bytes = base64.b64decode(img_b64)
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif "image" in request.files:
            pil = Image.open(request.files["image"].stream).convert("RGB")
        else:
            return jsonify({"error": "Görüntü bulunamadı"}), 400
    except Exception as e:
        return jsonify({"error": f"Görüntü okunamadı: {e}"}), 400

    inp  = _tf(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(_model(inp), dim=1)[0].cpu().numpy()

    top5_idx = probs.argsort()[::-1][:5]
    t1_idx   = int(top5_idx[0])
    t1_cls   = _classes[t1_idx]
    t1_conf  = float(probs[t1_idx])
    info     = DISEASE_INFO.get(t1_cls, {})

    # Grad-CAM
    inp_g = _tf(pil).unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam   = _gradcam.generate(inp_g, t1_idx)

    predictions = [{
        "label":      _classes[i],
        "name_tr":    DISEASE_INFO.get(_classes[i], {}).get("hastalik", _classes[i]),
        "plant_tr":   DISEASE_INFO.get(_classes[i], {}).get("bitki", "—"),
        "confidence": round(float(probs[i]), 4),
    } for i in top5_idx]

    tarih = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Grad-CAM görüntülerini bir kez hesapla
    cam_pure_b64    = gradcam_pure(cam)
    cam_overlay_b64 = gradcam_overlay(pil, cam)
    orig_b64        = to_b64(pil)

    return jsonify({
        # Frontend field names
        "predictions":   predictions,
        "disease_info":  info.get("aciklama", ""),
        "cause":         info.get("neden", ""),
        "prevention":    info.get("onlem", ""),
        "treatment":     info.get("tedavi", ""),
        "gradcam_image": cam_pure_b64,
        "overlay_image": cam_overlay_b64,
        # PDF / legacy fields
        "sinif":      t1_cls,
        "bitki":      info.get("bitki", "—"),
        "hastalik":   info.get("hastalik", t1_cls),
        "siddet":     info.get("siddet", "—"),
        "aciklama":   info.get("aciklama", ""),
        "neden":      info.get("neden", ""),
        "onlem":      info.get("onlem", ""),
        "tedavi":     info.get("tedavi", ""),
        "saglikli":   info.get("saglikli", False),
        "confidence": round(t1_conf * 100, 2),
        "original":   orig_b64,
        "gradcam":    cam_overlay_b64,
        "tarih":      tarih,
    })

@app.route("/pdf", methods=["POST"])
def pdf():
    try:
        data_j  = request.get_json(force=True)
        img_b64 = data_j.get("image", "")
        pil     = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Görüntü okunamadı: {e}"}), 400

    # Yeniden analiz çalıştır (PDF için tam veri hazırla)
    inp  = _tf(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(_model(inp), dim=1)[0].cpu().numpy()

    top5_idx = probs.argsort()[::-1][:5]
    t1_idx   = int(top5_idx[0])
    t1_cls   = _classes[t1_idx]
    t1_conf  = float(probs[t1_idx])
    info     = DISEASE_INFO.get(t1_cls, {})

    inp_g = _tf(pil).unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam   = _gradcam.generate(inp_g, t1_idx)

    top5 = [{
        "bitki":    DISEASE_INFO.get(_classes[i], {}).get("bitki", "—"),
        "hastalik": DISEASE_INFO.get(_classes[i], {}).get("hastalik", _classes[i]),
        "conf":     round(float(probs[i]) * 100, 2),
    } for i in top5_idx]

    pdf_data = {
        "sinif":      t1_cls,
        "bitki":      info.get("bitki", "—"),
        "hastalik":   info.get("hastalik", t1_cls),
        "siddet":     info.get("siddet", "—"),
        "aciklama":   info.get("aciklama", ""),
        "neden":      info.get("neden", ""),
        "onlem":      info.get("onlem", ""),
        "tedavi":     info.get("tedavi", ""),
        "saglikli":   info.get("saglikli", False),
        "confidence": round(t1_conf * 100, 2),
        "top5":       top5,
        "original":   to_b64(pil),          # sol: orijinal fotoğraf
        "gradcam":    gradcam_pure(cam),     # sağ: saf ısı haritası
        "tarih":      datetime.now().strftime("%d.%m.%Y %H:%M"),
    }

    buf = io.BytesIO(build_pdf(pdf_data))
    buf.seek(0)
    fn = f"teshis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name=fn)

# ── PDF ───────────────────────────────────────────────────────────────────────
def build_pdf(d):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2.2*cm, bottomMargin=2*cm)
    W = A4[0] - 4*cm

    # ── Stiller (Türkçe destekli font) ───────────────────────────────────────
    def ps(name, **kw):
        kw.setdefault("fontName", _FONT)
        return ParagraphStyle(name, **kw)

    title_s = ps("pdf_title",
                 fontSize=18, leading=22,
                 textColor=colors.HexColor("#1B5E20"),
                 spaceAfter=3, fontName=_FONT_BOLD)
    sub_s   = ps("pdf_sub",
                 fontSize=9.5, textColor=colors.HexColor("#558B2F"),
                 spaceAfter=10)
    h2_s    = ps("pdf_h2",
                 fontSize=12, fontName=_FONT_BOLD,
                 textColor=colors.HexColor("#2E7D32"),
                 spaceBefore=14, spaceAfter=7)
    body_s  = ps("pdf_body", fontSize=10, leading=16)
    lbl_s   = ps("pdf_lbl",  fontSize=9,
                 textColor=colors.HexColor("#757575"))
    val_s   = ps("pdf_val",  fontSize=11, fontName=_FONT_BOLD)
    hdr_s   = ps("pdf_hdr",  fontSize=10, fontName=_FONT_BOLD,
                 textColor=colors.white)
    foot_s  = ps("pdf_foot", fontSize=8,
                 textColor=colors.HexColor("#9E9E9E"),
                 alignment=TA_CENTER)
    cap_s   = ps("pdf_cap",  fontSize=8.5,
                 textColor=colors.HexColor("#558B2F"),
                 alignment=TA_CENTER)

    saglikli = d.get("saglikli", False)
    dur_renk = colors.HexColor("#2E7D32") if saglikli else colors.HexColor("#C62828")
    dur_text = "SAGLIKLI"    if saglikli else "HASTALIK TESPIT EDILDI"
    dur_s    = ps("pdf_dur", fontSize=11, fontName=_FONT_BOLD, textColor=dur_renk)

    story = [
        Paragraph("Bitki Hastaligi Tespit Raporu", title_s),
        Paragraph(f"Analiz Tarihi: {d.get('tarih', '—')}", sub_s),
        HRFlowable(width="100%", thickness=2,
                   color=colors.HexColor("#4CAF50"), spaceAfter=12),
    ]

    # ── Özet tablo ────────────────────────────────────────────────────────────
    rows = [
        [Paragraph("Bitki",    lbl_s), Paragraph(d.get("bitki",    "—"), val_s)],
        [Paragraph("Hastalik", lbl_s), Paragraph(d.get("hastalik", "—"), val_s)],
        [Paragraph("Guven",    lbl_s), Paragraph(f"%{d.get('confidence', 0):.1f}", val_s)],
        [Paragraph("Siddet",   lbl_s), Paragraph(d.get("siddet",   "—"), val_s)],
        [Paragraph("Durum",    lbl_s), Paragraph(dur_text, dur_s)],
    ]
    tbl = Table(rows, colWidths=[3.5*cm, W - 3.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#F1F8E9")),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#C8E6C9")),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.white, colors.HexColor("#FAFFFE")]),
    ]))
    story += [tbl, Spacer(1, 14)]

    # ── Görüntüler: sol=Orijinal, sağ=Grad-CAM ───────────────────────────────
    story.append(Paragraph("Goruntu Analizi", h2_s))
    iw = (W / 2) / cm - 0.6   # her görüntünün cm cinsinden genişliği

    def b64img(s, w_cm, h_cm):
        try:
            return RLImage(io.BytesIO(base64.b64decode(s)),
                           width=w_cm * cm, height=h_cm * cm)
        except Exception:
            return Paragraph("(goruntu yuklenemedi)", lbl_s)

    orig_img = b64img(d.get("original", ""), iw, iw)
    cam_img  = b64img(d.get("gradcam",  ""), iw, iw)

    img_tbl = Table(
        [[orig_img, cam_img]],
        colWidths=[W / 2, W / 2]
    )
    img_tbl.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))

    cap_tbl = Table(
        [[Paragraph("Orijinal Fotograf", cap_s),
          Paragraph("Grad-CAM Isi Haritasi", cap_s)]],
        colWidths=[W / 2, W / 2]
    )
    cap_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    story += [img_tbl, Spacer(1, 4), cap_tbl, Spacer(1, 14)]

    # ── Hastalık açıklaması, sebebi, önlemleri, tedavi ──────────────────────
    info_sections = [
        ("Hastalik Bilgisi",   d.get("aciklama", "—")),
        ("Hastalik Sebebi",    d.get("neden",    "—")),
        ("Alinacak Onlemler",  d.get("onlem",    "—")),
        ("Tedavi Onerileri",   d.get("tedavi",   "—")),
    ]
    for title, content in info_sections:
        if content and content != "—":
            story += [
                Paragraph(title, h2_s),
                Paragraph(content, body_s),
                Spacer(1, 8),
            ]
    story.append(Spacer(1, 6))

    # ── Top-5: güven skoruna göre sıralı, skor öne çıkarılmış ───────────────
    story.append(Paragraph("En Olasilikli 5 Tahmin (Guvene Gore)", h2_s))
    top5 = sorted(d.get("top5", []),
                  key=lambda x: x.get("conf", 0), reverse=True)
    if top5:
        header_row = [
            Paragraph("Guven Skoru", hdr_s),
            Paragraph("Hastalik",    hdr_s),
            Paragraph("Bitki",       hdr_s),
        ]
        rows5 = [header_row]
        for i, item in enumerate(top5):
            conf_val = item.get("conf", 0)
            # İlk satır (en yüksek) açık yeşil arka plan
            bg = colors.HexColor("#E8F5E9") if i == 0 else colors.white
            conf_para = ps(f"c{i}", fontSize=12, fontName=_FONT_BOLD,
                           textColor=colors.HexColor("#1B5E20") if i == 0
                           else colors.HexColor("#2E7D32"))
            rows5.append([
                Paragraph(f"%{conf_val:.2f}", conf_para),
                Paragraph(item.get("hastalik", "—"), body_s),
                Paragraph(item.get("bitki",    "—"), body_s),
            ])

        t5 = Table(rows5, colWidths=[3*cm, W - 6.5*cm, 3.5*cm])
        t5_style = [
            ("BACKGROUND",    (0, 0), (-1,  0), colors.HexColor("#2E7D32")),
            ("BACKGROUND",    (0, 1), (-1,  1), colors.HexColor("#E8F5E9")),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#C8E6C9")),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (0, 0), (0,  -1), "CENTER"),   # güven sütunu ortalı
        ]
        # Çift satır arkaplanı
        for r in range(2, len(rows5)):
            if r % 2 == 0:
                t5_style.append(("BACKGROUND", (0, r), (-1, r),
                                  colors.HexColor("#FAFFFE")))
        t5.setStyle(TableStyle(t5_style))
        story.append(t5)

    # ── Footer ────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#A5D6A7")),
        Spacer(1, 5),
        Paragraph(
            f"EfficientNet-B4 tabanli Bitki Hastaligi Tespit Sistemi"
            f" | {d.get('tarih', '')}",
            foot_s),
    ]

    doc.build(story)
    return buf.getvalue()

# ── Başlat ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    print("✓ http://127.0.0.1:5000 adresini tarayıcınızda açın\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
