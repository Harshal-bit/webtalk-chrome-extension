// script.js
document.addEventListener('DOMContentLoaded', function () {
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const responseDiv = document.getElementById('response');
    const pdfFileInput = document.getElementById('pdfFile');

    sendButton.addEventListener('click', async () => {
        const textToSend = userInput.value;
        const pdfFile = pdfFileInput.files[0];
        const formData = new FormData();
        formData.append('prompt', textToSend);
        formData.append('pdf', pdfFile);
        // Query the active tab in the current window
        chrome.tabs.query({ active: true, currentWindow: true }, async function (tabs) {
            // Since only one tab should be active and in the current window at once,
            // the returned variable should only have one entry.
            const activeTab = tabs[0];
            formData.append('url', activeTab.url);

            try {
                // Send the text to localhost server (adjust the URL as needed)
                const response = await fetch('http://localhost:8000', {
                    method: 'POST',
                    body: formData
                });
    
                const data = await response.json();
                responseDiv.appendChild(createResponseDiv(textToSend,data.result))
                
                //responseDiv.innerHTML = `Response from server: ${data.result}`;
            } catch (error) {
                responseDiv.innerHTML = `Error: ${error.message}`;
            }
        });

    });

    function createResponseDiv(question, response) {
        const container = document.createElement('div');
        container.className = 'container';
    
        const row = document.createElement('div');
        row.className = 'row';
    
        const col1 = document.createElement('div');
        col1.className = 'col-sm';
    
        const questionDiv = document.createElement('div');
        questionDiv.className = 'alert alert-primary';
        questionDiv.role = 'alert';
        questionDiv.textContent = question;
    
        const col2 = document.createElement('div');
        col2.className = 'col-sm';
    
        const responseDiv = document.createElement('div');
        responseDiv.className = 'alert alert-secondary';
        responseDiv.role = 'alert';
        responseDiv.textContent = response;
    
        col1.appendChild(questionDiv);
        col2.appendChild(responseDiv);
    
        row.appendChild(col1);
        row.appendChild(col2);
    
        container.appendChild(row);
    
        return container;
    }
});

