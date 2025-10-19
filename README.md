🧠 Mental Health Ring
💡 Inspiration

Mental health remains one of the most pressing issues of our time. While stress tracking exists, it often fails to address the deeper, underlying concerns of mental wellness. Every day, people continue to neglect their mental health — despite scientific evidence showing its growing importance. Our project takes a step forward in bridging that gap through real-time biometric monitoring and personalized feedback.

⚙️ What It Does

The Mental Health Ring uses an ESP-32 microcontroller equipped with a heartbeat sensor and a temperature sensor to collect biometric data.

These readings are sent to a Flask REST API, which forwards the data to a PyTorch model.

The model, powered by Gradient Optimization, CNNs, and carefully engineered feature sets, predicts the severity of stress or irregular heartbeat patterns.

Finally, the Flask API returns the model output to a Next.js frontend, where users can view their results and receive customized breathing exercises to improve their mental state.

This end-to-end system enables real-time monitoring and personalized feedback — promoting better mental health management.

🛠️ How We Built It

Autodesk Fusion 360 – Designed and modeled the hardware casing.

Arduino (C++) – Programmed the ESP-32 for sensor data collection.

PyTorch, Flask, Scikit-learn – Built and deployed the CNN model for pattern detection.

Next.js, Tailwind CSS – Created the responsive, accessible frontend interface.

🚧 Challenges We Ran Into

Model accuracy: Achieving reliable results required over 6 hours of tuning to combat overfitting, misalignment, and limited feature data.

Backend connectivity: Coordinating multiple devices over Wi-Fi and ensuring stable Flask API communication proved difficult.

Frontend optimization: Designing a user-friendly and performant interface required constant testing and iteration.

🏆 Accomplishments We’re Proud Of

Achieved 100% accuracy on our final 10% test dataset.

Established a stable Flask API connection over a mobile hotspot with strong data throughput.

Fully implemented and polished all UI features for an intuitive user experience.

📚 What We Learned

Integrating multiple Arduino sensors effectively.

Building accurate ML models with limited data.

Designing a hands-free product that provides valuable insights without constant human intervention.

🚀 What’s Next for Mental Health Ring

Further optimizing API response times for smoother real-time performance.

Enhancing user-specific recommendations by expanding the dataset and personalization algorithms.

Exploring additional sensors (e.g., GSR, oxygen levels) for more comprehensive mental health tracking.
