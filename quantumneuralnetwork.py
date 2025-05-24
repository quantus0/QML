from qiskit import BasicAer
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import TwoLocal
import torch
import numpy as np

algorithm_globals.random_seed = 42

ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz', reps=1, entanglement='linear')

qnn = SamplerQNN(
    circuit=ansatz,
    input_params=ansatz.parameters[:2],
    weight_params=ansatz.parameters[2:]
)

model = TorchConnector(qnn)

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

with torch.no_grad():
    preds = model(X).argmax(dim=1)
    accuracy = (preds == y).float().mean()
    print(f"Training accuracy: {accuracy.item()*100:.2f}%")
