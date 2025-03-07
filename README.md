

<h1>🔐 Vault 17 - Forensic Signature Verification</h1>

<h2>📌 Project Overview</h2>
<p>Vault 17 is under attack—highly sophisticated forgeries are infiltrating classified documents. As part of the <strong>Vault 17 Forensic AI Division (V17FAD)</strong>, your mission is to develop an AI-based signature verification system to distinguish between real and forged signatures.</p>

<h2>🎯 Objectives</h2>
<ul>
    <li>Analyze the dataset and extract meaningful features.</li>
    <li>Train a binary classifier using classical Machine Learning (no Deep Learning).</li>
    <li>Evaluate the model’s performance and ensure high interpretability.</li>
    <li>Test the system against real-world signature samples.</li>
</ul>

<h2>🚀 Bonus Challenge</h2>
<p>A new wave of forgeries has emerged—style-transferred signatures that mimic real handwriting patterns. To stay ahead of forgers, you will:</p>
<ul>
    <li>Use <strong>Neural Style Transfer (NST)</strong> to generate advanced forgeries.</li>
    <li>Test the model against these synthetic forgeries.</li>
    <li>Document findings in a forensic AI report.</li>
</ul>

<h2>📂 Dataset</h2>
<ul>
    <li><strong>Real Signatures:</strong> <code>original_signs/</code> (1321 images)</li>
    <li><strong>Forged Signatures:</strong> <code>forged_signs/</code> (1321 images)</li>
    <li>Dataset Source: <a href="https://www.kaggle.com/datasets/shreelakshmigp/cedardataset/data" target="_blank">CEDAR Dataset</a></li>
</ul>

<h2>🛠️ Installation</h2>
<p>Clone the repository and set up the virtual environment:</p>

<pre><code>git clone https://github.com/your-username/vault17-signature-verification.git
cd vault17-signature-verification
python -m venv venv
source venv/bin/activate  # (For Mac/Linux)
venv\Scripts\activate  # (For Windows)
pip install -r requirements.txt
</code></pre>

<h2>📜 How to Run</h2>
<h3>1️⃣ Preprocess the Data</h3>
<pre><code>python data_preprocessing.py</code></pre>

<h3>2️⃣ Train the Model</h3>
<pre><code>python train_model.py</code></pre>

<h3>3️⃣ Test the Model on a New Signature</h3>
<pre><code>python test_signature.py</code></pre>

<h2>📊 Model Performance</h2>
<p>The classifier achieved <strong>100% accuracy</strong> on the test set.</p>
<img src="confusion_matrix.png" alt="Confusion Matrix" width="500">

<h2>📜 Results</h2>
<p>Model Performance:</p>
<ul>
    <li><strong>Accuracy:</strong> 100%</li>
    <li><strong>Precision:</strong> 1.00</li>
    <li><strong>Recall:</strong> 1.00</li>
</ul>

<h2>📝 Future Improvements</h2>
<ul>
    <li>Test the model against unseen real-world signature samples.</li>
    <li>Implement adversarial training to improve robustness.</li>
    <li>Apply Neural Style Transfer (NST) for advanced forgery detection.</li>
</ul>

<h2>📌 License</h2>
<p>This project is licensed under the <strong>MIT License</strong>.</p>

<h2>📞 Contact</h2>
<p>For inquiries or contributions, feel free to reach out via GitHub.</p>

</body>
</html>
