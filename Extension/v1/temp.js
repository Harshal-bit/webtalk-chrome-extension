document.addEventListener('DOMContentLoaded', function () {
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const responseDiv = document.getElementById('response');
    const pdfFileInput = document.getElementById('pdfFile');

    sendButton.addEventListener('click', async () => {
        const textToSend = userInput.value;
        const pdfFile = pdfFileInput.files[0];

        // Create a FormData object to send both text and file
        const formData = new FormData();
        formData.append('prompt', textToSend);
        formData.append('pdf', pdfFile);

        try {
            // Send the text and PDF file to localhost server
            const response = await fetch('http://localhost:8000/pdf', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            responseDiv.innerHTML = `Response from server: ${data.result}`;
        } catch (error) {
            responseDiv.innerHTML = `Error: ${error.message}`;
        }
    });
});