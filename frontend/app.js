const express = require("express");
const cors = require("cors");
const app = express();
const port = 3000;

app.use(cors());

app.use(express.static("public"));

app.get("/", (req, res, next) => {
  res.sendFile(__dirname + "/pages/index.html");
});

app.get("/models", (req, res, next) => {
  res.sendFile(__dirname + "/pages/models.html");
});

app.use((req, res, next) => {
  res.sendFile(__dirname + "/pages/404.html");
});

app.listen(port, () => {
  console.log(`Server listening at port: ${port}...`);
});
