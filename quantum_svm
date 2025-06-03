from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Generate toy dataset
X, y = make_classification(n_samples=40, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Quantum feature map and kernel
feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

# Kernel matrix for SVM
kernel_matrix_train = quantum_kernel.evaluate(x_vec=X_train)
kernel_matrix_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

# Train and test SVM
svm = SVC(kernel='precomputed')
svm.fit(kernel_matrix_train, y_train)
y_pred = svm.predict(kernel_matrix_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"QSVM classification accuracy: {accuracy * 100:.2f}%")
