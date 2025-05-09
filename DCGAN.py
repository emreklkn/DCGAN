# -*- coding: utf-8 -*-
"""
Created on Fri May  9 04:19:21 2025

@author: emrek
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparametreler
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
learning_rate = 0.0002

# MNIST veri seti
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist = torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# Ayrıştırıcı (Discriminator)
"""AYRIŞTIRICI BU HER BİR FOTOĞRAFI KONTROL EDİYOR VE BUNU GERİ BİLDİRİYOR DOĞRU FOTOĞRAFMI YANLIŞ FOTOĞRAFMI"""
"""DİSCRİMİNATİOR U ÇOK İYİ EĞİTİRSEK GENERATOR HİÇBİR ZAMAN DOĞRU FOTOĞRAFI ÜRETEMEZ BELİRLİ SEVİYEDE TUTACAĞIZ"""
"""Leaky ReLU, Discriminator için daha stabil ve güvenli bir seçimdir."""
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
).to(device)

# Üreteç (Generator)
"""SAHTE FOTOĞRAF ÜRETEREK DİSCRİMİNATOR U KANDIRMAYA ÇALIŞICAK"""
"""ReLU, Generator için daha etkili olabilir çünkü modelin sınırlı bir alanı keşfetmesine yardımcı olur."""
"""3 KATMAN VAR 2SİNDE RELU DİĞERİNDE TANH AKTİVASYON FONK KULLANILMIS"""
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),    
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
).to(device)

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
"""BİRÇOK MODELDEKİ GİBİ HIZLI VE İYİ OLAN ADAM OPTİMİZEER KULLANILDU"""

# Eğitim
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Gerçek ve sahte etiketler
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Gerçek görüntüler
        images = images.reshape(batch_size, -1).to(device)# NORMAL RESME CEVİRME

        # Ayrıştırıcı için kayıp ve geri yayılım
        outputs = D(images) # AYRIŞTIRICI AĞA FOTOĞRAF VERİLİYOR HER EPOCHTA VE BUNU ÖĞRENİYOR 38.SATIRDA YUKARDA D()
        d_loss_real = criterion(outputs, real_labels)#Discriminator'ın gerçek görüntüleri doğru sınıflandırma kaybı hesaplanır.
        real_score = outputs

        # Sahte görüntüler
        z = torch.randn(batch_size, latent_size).to(device)# RASTGELE FOTOĞRAF ÜRETİLMESİ İÇİN BATCH SİZE VE LATENT SİZE ÜRETİLİR
        fake_images = G(z) # GENERATORDA FAKE İMAGE TASARLANIR
        outputs = D(fake_images)# FAKE İMAGE AYRISTIRICIYA SOKULUR 
        d_loss_fake = criterion(outputs, fake_labels) # KAYIP DEĞERLERİ HESAPLANIR KRİTERE GÖRE YUKARDA 71 VE 72 SATIRDAKİ SINIFLANDIRILIR
        fake_score = outputs

        # Toplam ayrıştırıcı kaybı
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Üreteç için kayıp ve geri yayılım
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    # Her 20 epoch'ta bir örnek görüntü göster
    if (epoch+1) %20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
        fake_images = fake_images.reshape(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(fake_images, nrow=10, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
        plt.show()
