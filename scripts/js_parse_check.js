const fs = require('fs');
const code = fs.readFileSync(process.argv[2], 'utf8');
try {
  new Function(code);
  console.log('OK: ' + process.argv[2] + ' (' + code.length + ' bytes)');
} catch (e) {
  console.log('FAIL: ' + e.message);
}
