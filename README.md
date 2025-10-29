# IBM Qiskit

## Problem Statement

**Quantum Reservoir Computing for Renewable Energy Forecasting**

**Background**: Renewable energy sources such as wind and solar exhibit high variability due to environmental fluctuations. Accurate forecasting of their output is crucial for sustainable grid management. Quantum Reservoir Computing (QRC) integrates quantum dynamics with classical regression, providing an alternative to recurrent networks for modeling complex, nonlinear time series.

**Task**: Participants can apply Quantum Reservoir Computing to predict renewable energy generation based on historical weather features like wind speed and irradiance. The results can be compared to a classical RNN to evaluate efficiency and stability.

## System Architecture for the Quantum Segment

**Project Title:** Hybrid Quantum-Classical System for Renewable Energy Source Classification

**Objective:** To build a proof-of-concept system using PennyLane that processes time-series weather data to recommend the most viable renewable energy source (e.g., Solar, Wind, Hydro) for a given location. The system will use a Quantum Reservoir Computer (QRC) for feature extraction and a Quantum Support Vector Classifier (QSVC) for classification.

**My Skill Level:** I am a beginner in PennyLane and Quantum Machine Learning. Please provide clear, modular code with detailed comments explaining each step, function, and quantum concept.

**Core Technologies:**

* **PennyLane:** For all quantum circuits (QRC and QSVC).  
* **Scikit-learn:** For classical data processing, SVC (Support Vector Classifier), and metrics.  
* **NumPy:** For all numerical data and vector operations.  
* **Matplotlib:** For visualization.

### **Step-by-Step Implementation Plan**

Please help me build this system in a single Python file.

#### **Phase 1: Setup and Mock Data Generation**

1. **Imports:** Import all necessary libraries (PennyLane, qml.device, QNode, NumPy, Matplotlib, and sklearn).  
2. **Global Constraint:** Define a global boolean variable HAS\_RIVER\_IN\_AREA \= True. We will use this later to constrain the model.  
3. **Mock Data Function:** Create a function generate\_mock\_data(days=365).  
   * This function should return a NumPy array of shape (days, 3\) representing daily weather features: \[solar\_irradiance, wind\_speed, water\_flow\].  
   * It should also return a corresponding array of labels (days,).  
   * **Label Logic:** Create a simple function get\_label(features) that returns a string label.  
     * If solar\_irradiance is the dominant feature, return "Solar".  
     * If wind\_speed is dominant, return "Wind".  
     * If water\_flow is dominant AND HAS\_RIVER\_IN\_AREA is True, return "Hydro".  
     * If water\_flow is dominant BUT HAS\_RIVER\_IN\_AREA is False, return "None" (or the next dominant type).  
     * Otherwise, return "Mixed".  
   * Generate 365 days of data.

#### **Phase 2: Module 1 \- The Quantum Reservoir Computer (QRC)**

This module's goal is to take a day's weather features and the previous day's reservoir state to generate a new, high-dimensional reservoir state.

1. **Define Reservoir Circuit:**  
   * Create a PennyLane quantum device (e.g., default.qubit with n\_qubits \= 5).  
   * Create a QNode qrc\_circuit that implements the "Quantum Dynamics" part. This circuit should take the previous reservoir state (as parameters for rotation gates) and apply a fixed, entangling layer (e.g., qml.StronglyEntanglingLayers).  
2. **Define Input Encoding:**  
   * Create a simple classical function encode\_input(features, n\_qubits). This function will map our 3 weather features to the n\_qubits of the reservoir. (e.g., repeating and scaling features to create an n\_qubit-sized vector).  
3. **Create the Reservoir "Driver" Function:**  
   * Create a function run\_qrc(daily\_features\_list).  
   * This function should:  
     * Initialize a starting reservoir state (e.g., zeros).  
     * Create a list to store the final reservoir state for each day.  
     * Loop through each day's features in daily\_features\_list:  
       * Combine the encoded input features with the previous reservoir state.  
       * Run the qrc\_circuit with these combined parameters.  
       * Measure the expectation values (e.g., qml.expval(qml.PauliZ(i)) for each qubit) to get the *new* reservoir state.  
       * Store this new state.  
   * This function should return a list of high-dimensional vectors, where each vector represents the "condensed" time-series information for that day. This aligns with **Step 1** of our plan.

#### **Phase 3: Module 2 \- Classical Condensation (Vector-to-Image)**

As per **Step 3** of our plan, we need to convert the reservoir's output vector into an "image" (2D matrix) for the QSVC.

1. **Gramian Angular Field (GAF) Function:**  
   * Please provide a simple Python/NumPy function vector\_to\_image(vector).  
   * This function will implement a **Gramian Angular Summation Field (GASF)**.  
   * It should:  
     1. Take the 1D reservoir state vector.  
     2. Rescale it to be in the \[-1, 1\] range.  
     3. Calculate the GAF matrix.  
     4. Return this 2D matrix (our "image").  
   * *(Optional Visualization):* Show how to use matplotlib.pyplot.imshow to visualize one of these "images." This helps with **Step 2** of our plan.

#### **Phase 4: Module 3 \- The Quantum Support Vector Classifier (QSVC)**

This module will classify the "images" from Phase 3, fulfilling **Step 4** of our plan.

1. **Prepare Data for QSVC:**  
   * Run all 365 days of mock data through run\_qrc (Phase 2\) to get 365 reservoir vectors.  
   * Run all 365 vectors through vector\_to\_image (Phase 3\) to get 365 "images."  
   * Flatten these images (e.g., if 5x5, flatten to 25x1) so they can be used as feature vectors by the QSVC.  
   * Get the corresponding labels from our mock data function.  
   * **Apply Constraint:** If HAS\_RIVER\_IN\_AREA \= False, filter out *all* samples labeled "Hydro". The QSVC should never be trained on this class.  
   * Split the data into a training (X\_train, y\_train) and testing (X\_test, y\_test) set.  
2. **Define the Quantum Kernel:**  
   * Define a quantum feature map circuit (e.g., qml.AngleEmbedding) as a QNode. This circuit will take one flattened "image" vector as its features argument.  
   * This QNode will be used by PennyLane to calculate the similarity between data points.  
3. **Create and Train the SVC (using the Quantum Kernel):**  
   * Use the qml.kernels.kernel\_matrix function to compute the kernel matrix for the training data. This function will repeatedly run your feature map QNode.  
     * kernel\_train \= qml.kernels.kernel\_matrix(X\_train, X\_train, kernel=your\_feature\_map\_qnode, ...)  
   * Initialize a classical sklearn.svm.SVC with kernel='precomputed' and probability=True.  
   * Train the SVC on the *precomputed training kernel matrix* (kernel\_train) and the training labels (y\_train).  
4. **Evaluate the Model:**  
   * Compute the test kernel matrix, which compares *test* data to the *training* data:  
     * kernel\_test \= qml.kernels.kernel\_matrix(X\_test, X\_train, kernel=your\_feature\_map\_qnode, ...)  
   * Use svc.predict(kernel\_test) and svc.predict\_proba(kernel\_test) to get results for the test set.  
   * Print a classification report.

#### **Phase 5: Module 4 \- Annual Analysis and Recommendation**

This module implements the final logic from **Steps 5 and 6** of our plan.

1. **Run Full-Year Simulation:**  
   * Take the full 365-day dataset (we'll call it X\_full).  
   * Run it through the first two parts of the pipeline:  
     * qrc\_vectors \= run\_qrc(X\_full)  
     * image\_vectors \= \[vector\_to\_image(v) for v in qrc\_vectors\]  
     * X\_full\_flat \= \[img.flatten() for img in image\_vectors\]  
   * **Important:** To get probabilities, you must compute the kernel comparing the full dataset to the *original training data* (X\_train) that the SVC was trained on.  
     * kernel\_full\_year \= qml.kernels.kernel\_matrix(X\_full\_flat, X\_train, kernel=your\_feature\_map\_qnode, ...)  
   * Get the daily probabilities:  
     * daily\_probabilities \= qsvc.predict\_proba(kernel\_full\_year)  
   * Store these daily probability scores. These are the "thresholds" from **Step 5**.  
2. **Implement Recommendation Logic:**  
   * Create a final analysis function.  
   * **Tally Scores:** Calculate the *total* probability score for each energy type over the 365 days.  
   * **Find Best Timeframes:** Find the month (or 30-day window) with the highest average score for each energy type.  
   * **Generate Recommendation:**  
     * Find the energy type with the highest total score (e.g., "Solar").  
     * Check if the second-highest type's score is close (e.g., within 90% of the top score).  
     * Print a final summary:  
       * "Primary Recommendation: \[Top Energy Type\] (Peak Season: \[Best Month\])"  
       * If a second type is close: "Secondary Recommendation: \[Second Energy Type\] (Peak Season: \[Best Month\])"