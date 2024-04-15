const { exec } = require('child_process');

const pythonScript = 'multimean.py';

exec(`python ${pythonScript}`, (error, stdout, stderr) => {
    if (error) {
	console.error(`${error}`);
        return;
    }
    console.log(`${stdout}`);
});

