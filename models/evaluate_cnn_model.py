def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    print(f'Test Accuracy: {acc:.2f}%')
    return all_labels, all_preds

# Train the model
train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

# Evaluate the model
y_true, y_pred = evaluate_model(model, test_loader, device)
