async function trainModel() {
    const csvContent = document.getElementById('csvInput').value;
    const targetCol = document.getElementById('targetCol').value;
    const modelType = document.getElementById('modelType').value;
    const testSize = parseFloat(document.getElementById('testSize').value);
    const outputDiv = document.getElementById('output');
    const trainBtn = document.getElementById('trainBtn');

    if (!csvContent || !targetCol) {
        alert("Please provide CSV data and the target column name.");
        return;
    }

    trainBtn.innerText = "Training...";
    trainBtn.disabled = true;
    outputDiv.innerText = "";

    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                csv_data: csvContent,
                target_col: targetCol,
                model_type: modelType,
                test_size: testSize
            })
        });

        const result = await response.json();
        trainBtn.innerText = "Train Model";
        trainBtn.disabled = false;

        if (result.error) {
            outputDiv.innerText = "Error: " + result.error;
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('predictSection').style.display = 'none';
        } else {
            let outputText = `Model Trained Successfully!\n\n`;
            outputText += `Algorithm: ${modelType}\n`;
            outputText += `${result.metric_name}: ${result.metric_value.toFixed(4)}\n`;
            outputText += `Intercept: ${result.intercept.toFixed(4)}\n`;
            outputText += `Coefficients: ${JSON.stringify(result.coef)}`;

            outputDiv.innerText = outputText;
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('predictSection').style.display = 'block';
        }

    } catch (error) {
        trainBtn.innerText = "Train Model";
        trainBtn.disabled = false;
        outputDiv.innerText = "Network Error: " + error.message;
        document.getElementById('resultsSection').style.display = 'block';
    }
}

async function predict() {
    const featureInput = document.getElementById('featureInput').value;
    const resultDiv = document.getElementById('predictionResult');

    if (!featureInput) {
        alert("Please enter feature values.");
        return;
    }

    // Convert comma string to array of numbers
    const features = featureInput.split(',').map(item => parseFloat(item.trim()));

    // Basic validation
    if (features.some(isNaN)) {
        alert("Invalid input: Ensure all feature values are numbers.");
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                features: features
            })
        });

        const result = await response.json();

        if (result.error) {
            resultDiv.innerText = "Error: " + result.error;
        } else {
            resultDiv.innerText = "Prediction: " + result.prediction.toFixed(4);
        }

    } catch (error) {
        resultDiv.innerText = "Network Error: " + error.message;
    }
}
