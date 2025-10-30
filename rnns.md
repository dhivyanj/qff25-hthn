# Recurring neural networks

* Recurrent Neural Networks (RNNs) are used in any application that involves sequential data, where the order of information matters. Their internal memory allows them to process and understand context in sequences like text, speech, and time-series data.
* Only RNNs are capable of Time-Series Analysis & Forecasting where they are specialized for finding temporal or sequential patterns (patterns over time or in a specific order)
* At their core, Recurrent Neural Networks (RNNs) work by having a loop that allows information to be passed from one step of the network to the next.

## Types of RNNs

* **Long Short-Term Memory (LSTM):** LSTMs are the most popular and widely used RNN architecture. They are explicitly designed to remember information for long periods.
* **Gated Recurrent Unit (GRU):** A GRU is a newer and simpler alternative to the LSTM. It provides similar performance on many tasks but is more computationally efficient.
* **Bidirectional RNNs (BRNNs):** *(beyond the scope of this problem)* This architecture processes the input sequence in two directions: once from beginning to end (forward) and once from end to beginning (backward).