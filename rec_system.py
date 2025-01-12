from google.colab import files
!pip install -q kaggle
files.upload() # установка токена

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets list -s "movielens100k"
!kaggle datasets download -d "rajmehra03/movielens100k"

!unzip movielens100k.zip

!rm movielens100k.zip

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

ratings = pd.read_csv("./ratings.csv").drop("timestamp", axis=1)
def tabular_preview(ratings, n=15):
  user_groups = ratings.groupby("userId")["rating"].count()
  top_users = user_groups.sort_values(ascending=False)[:n]

  movies_groups = ratings.groupby("movieId")["rating"].count()
  top_movies = movies_groups.sort_values(ascending=False)[:n]

  top = (
    ratings.
    join(top_users, rsuffix='_r', how='inner', on='userId').
    join(top_movies, rsuffix='_r', how='inner', on='movieId'))

  return pd.crosstab(top.userId, top.movieId, top.rating, aggfunc=np.sum)

class MovieDataset(Dataset):
  def __init__(self, users, movies, ratings):
    self.users = users
    self.movies = movies
    self.ratings = ratings

  def __len__(self):
    return len(self.users)

  def __getitem__(self, item):
    users = self.users[item]
    movies = self.movies[item]
    ratings = self.ratings[item]

    return {
        "users": torch.tensor(users),
        "movies": torch.tensor(movies),
        "ratings": torch.tensor(ratings)
    }
    class RecModel(nn.Module):
  def __init__(self, n_users, n_movies):
    super().__init__()
    self.users_emb = nn.Embedding(n_users, 32)
    self.movies_emb = nn.Embedding(n_movies, 32)

    self.linear = nn.Linear(64, 1)

  def forward(self, users, movies):
    users_emb = self.users_emb(users)
    movies_emb = self.movies_emb(movies)
    users_movies = torch.cat([users_emb, movies_emb], 1)

    output = self.linear(users_movies)
    return output
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lbl_users = LabelEncoder()
lbl_movies = LabelEncoder()

ratings.userId = lbl_users.fit_transform(ratings.userId.values)
ratings.movieId = lbl_movies.fit_transform(ratings.movieId.values)

X_train, X_val = train_test_split(ratings, test_size=0.1, random_state=42, stratify=ratings.rating)

train_dataset = MovieDataset(movies=X_train["movieId"].values,
                             users=X_train["userId"].values,
                             ratings=X_train["rating"].values)

test_dataset = MovieDataset(movies=X_val["movieId"].values,
                             users=X_val["userId"].values,
                             ratings=X_val["rating"].values)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=2)

model = RecModel(
    n_users = len(lbl_users.classes_),
    n_movies = len(lbl_movies.classes_)
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

iter_loss = 0
total_losses = []
total_losses.clear()

EPOCHS = 1
step_print, step_plot = 5000, 5000
count_step = 0


for epoch in range(EPOCHS):
  model.train()
  for data in train_dataloader:

    users = data["users"].to(DEVICE)
    movies = data["movies"].to(DEVICE)

    outputs = model(users, movies)
    rating = data["ratings"].view(len(users), -1).to(torch.float32).to(DEVICE)

    loss = criterion(outputs, rating)

    iter_loss += loss.sum().item()
    count_step += len(users)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if count_step % step_plot == 0:
      avg_loss = iter_loss / (len(users) * step_plot)
      print("epoch: {}; step {} and avg_loss: {}".format(epoch+1, count_step, avg_loss))
      total_losses.append(avg_loss)
      iter_loss = 0

plt.figure(figsize=(15, 6))
plt.plot(range(len(total_losses)),
         total_losses,
         linestyle="-"
         )
plt.xlabel("Iter Step")
plt.ylabel("Loss")
plt.title("RecModel")
plt.show()

from sklearn.metrics import mean_squared_error

model_outputs = []
targets = []

model.eval()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
    model_output = model(data["users"], data["movies"])
    model_outputs.append(model_output.sum().item() / len(data["users"]))

    target = data["ratings"]
    targets.append(target.sum().item() / len(data["users"]))

    print(f"Iteration: {i+1}; Model_output: {model_output}; Raiting: {target}")

print(f"Mean Square Error: {mean_squared_error(targets, model_outputs)}")
