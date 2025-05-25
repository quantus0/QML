from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

algorithm_globals.random_seed = 42

feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')

quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=None)

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

qsvc = QSVC(quantum_kernel=quantum_kernel)

qsvc.fit(X_train, y_train)

y_pred = qsvc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test set accuracy: {accuracy * 100:.2f}%")
