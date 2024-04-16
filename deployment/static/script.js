document.getElementById('csvFileInput').addEventListener('change', handleFileSelect);

function handleFileSelect(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const csv = e.target.result;
        Papa.parse(csv, {
            complete: function(results) {
                displayCSV(results.data);
            }
        });
    };

    reader.readAsText(file);
}

function displayCSV(data) {
    const table = document.createElement('table');

    data.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    // Clear previous data and append new table
    const csvDataDiv = document.getElementById('csvData');
    csvDataDiv.innerHTML = '';
    csvDataDiv.appendChild(table);
}