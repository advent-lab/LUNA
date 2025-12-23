# LUNA: LUT-Based Neural Architecture for Fast and Low-Cost Qubit Readout

![LUNA System Diagram](https://github.com/malifarooq/LUNA/blob/main/readme/Integrator_Logicnet.png)

LUNA is a fast and efficient superconducting qubit readout accelerator that combines low-cost integrator-based preprocessing with Look-Up Table (LUT) based neural networks for classification. The architecture uses simple integrators for dimensionality reduction with minimal hardware overhead, and employs LogicNets (DNNs synthesized into LUT logic) to drastically reduce resource usage while enabling ultra-low-latency inference. We integrate this with a differential evolution based exploration and optimization framework to identify high-quality design points. Our results show up to a **10.95× reduction in area** and **30% lower latency** with little to no loss in fidelity compared to the state-of-the-art. 

### Using the flow:
`logicnets\experiments\binary_classification` contains the code and instruction for setting up the differential evolution script and subsequent model training

`preprocessor` contains the adder tree templates

### The final designs
`quantum-net-final` contains final designs. 

