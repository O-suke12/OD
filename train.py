from tqdm import tqdm

def train(epochs, model, optimizer, loss_fn, train_loader):
    for epoch in range(epochs):

      for batch, label in tqdm(train_loader):
        model.train()
        y_pred = model(batch)
        loss = loss_fn(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # with torch.inference_mode():
      #   test_loss = 0
      #   for batch, label in tqdm(test_dataloader):
      #     test_pred = model(batch)
      #     loss = loss_fn(test_pred, label)
      #     test_loss += loss
      #   print(test_loss)