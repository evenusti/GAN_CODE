
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parametri generali
latent_dim = 10  # Dimensione del rumore di input
num_points = 24  # Punti per una curva (24 ore)
batch_size = 32  # Batch size
learning_rate = 0.00002  # Tasso di apprendimento
epochs = 4000  # Numero di epoche

# Caricamento del dataset da file Excel
file_path = "DATASET_PROVA.xlsx"
data = pd.read_excel(file_path)  # Qui serve openpyxl per leggere il file

# Verifica della forma dei dati
print("Forma dei dati originali:", data.shape)

# Creazione del torch tensor
curves = data.pivot_table(index=None, columns="ORA", values="PRODUZIONE").values
curves = torch.tensor(curves, dtype=torch.float32)

# Verifica della dimensione
print(f"Dimensione dei dati reali: {curves.shape}")

# Normalizzazione dei dati tra 0 e 1
curves = (curves - curves.min()) / (curves.max() - curves.min())

# Calcola la curva media reale per la visualizzazione alla fine dell'addestramento
average_curve = curves.mean(dim=0).numpy()


# Funzione per generare batch di dati reali
def generate_real_data_from_excel(batch_size, curves):
    indices = torch.randint(0, curves.shape[0], (batch_size,))
    real_data = curves[indices]
    return real_data


# Modello generatore
class Generator(nn.Module):
    def __init__(self, latent_dim, num_points):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_points),
        )

    def forward(self, z):
        return self.model(z)


# Modello discriminatore
class Discriminator(nn.Module):
    def __init__(self, num_points):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_points, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Inizializzazione dei modelli
G = Generator(latent_dim, num_points)
D = Discriminator(num_points)

# Ottimizzatori e funzione di perdita
criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

# Lista per salvare le curve generate ogni 100 epoche
generated_curves_all = []

# Addestramento della GAN
for epoch in range(epochs):
    # Generazione batch di dati reali
    real_curves = generate_real_data_from_excel(batch_size, curves)
    real_labels = torch.ones(batch_size, 1)

    # Generazione dei dati falsi
    z = torch.randn(batch_size, latent_dim)
    fake_curves = G(z)
    fake_labels = torch.zeros(batch_size, 1)

    # Allenamento del discriminatore
    D_real = D(real_curves)
    D_fake = D(fake_curves.detach())

    d_loss_real = criterion(D_real, real_labels)
    d_loss_fake = criterion(D_fake, fake_labels)
    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # Allenamento del generatore
    D_fake = D(fake_curves)
    g_loss = criterion(D_fake, real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    # Visualizzazione dei risultati
    if epoch == 0:
        print(f"Epoch {epoch}: Dimensione dei dati reali nel batch: {real_curves.shape}")
        print(f"Epoch {epoch}: Dimensione dei dati generati: {fake_curves.shape}")

    # Salvataggio della curva generata ogni 100 epoche
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        z = torch.randn(1, latent_dim)
        generated_curve = G(z).detach().numpy()[0]
        generated_curves_all.append(generated_curve)

        # Visualizzazione della curva generata
        plt.plot(range(num_points), generated_curve, label="Generated", color="blue")
        plt.plot(range(num_points), average_curve, label="Real Average", color="orange")
        plt.legend()
        plt.title(f"Epoch {epoch}: Generated vs Real Average Curve")
        plt.show()

# Salvataggio delle curve generate in file CSV
output_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'generated_curves.csv')
df = pd.DataFrame(generated_curves_all, columns=[f"Hour_{i + 1}" for i in range(num_points)])
df.index.name = "Epoch (x100)"
df.to_csv(output_path, index=True)

print(f"Le curve generate sono state salvate in '{output_path}'!")
