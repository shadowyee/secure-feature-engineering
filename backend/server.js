const express = require('express');
const multer = require('multer');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

const { exec } = require('child_process');

const pythonScript = 'multimean.py';

// 配置 Multer 中间件，用于处理文件上传
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/'); // 文件保存的目录，可以根据需要修改
  },
  filename: function (req, file, cb) {
    // cb(null, file.originalname); // 使用原始文件名作为保存的文件名
    cb(null, "test.txt"); // 使用"test.txt作为保存的文件名
  }
});

const upload = multer({ storage: storage });

// 配置静态文件目录，用于访问上传的文件
app.use(express.static(path.join(__dirname, 'uploads')));
app.use(cors());

// 设置路由，用于处理文件上传请求
app.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No files were uploaded.');
  }

  exec(`python ${pythonScript}`, (error, stdout) => {
    if (error) {
        return res.status(500).json({ error: 'An error occurred while running Python script.' });
    }
    console.log(`${stdout}`);
    
    res.json({ result: stdout });
  });
  
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

