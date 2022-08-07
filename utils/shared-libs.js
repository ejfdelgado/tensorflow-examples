const fs = require('fs');

// Get executable argument
let executable = null;
let output_name = "shared-libs.zip";
process.argv.forEach(function (val, index, array) {
    if (index == 2) {
        executable = val;
    }
    if (index == 3) {
        output_name = val;
    }
});
if (executable == null) {
    console.log('define the executable name')
    return;
}

async function runExecutable(path, arguments) {
    const spawnSync = require('child_process').spawnSync;
    const child = spawnSync(path, arguments, { encoding: 'utf8' });
    return child.stdout;
}

function copyFile(origin, destinationFolder) {
    const fileName = /([^\/]+)$/.exec(origin)[1];
    return new Promise((resolve, reject) => {
        fs.copyFile(origin, destinationFolder + "/" + fileName, (err) => {
            if (err) {
                reject(err)
            } else {
                resolve();
            }
        });
    });
}

async function analize() {
    const response = await runExecutable("ldd", [executable]);
    const lineas = response.split('\n');
    const pathFiles = [];
    // Get file paths
    for (let i = 0; i < lineas.length; i++) {
        const linea = lineas[i];
        const partes = /=>(.+)$|(\/.+\s\()/igm.exec(linea);
        //console.log(JSON.stringify(partes));
        if (partes != null) {
            let path = partes[1] ? partes[1] : partes[2];
            //console.log(path);
            path = path.replace(/\s+\(.*/ig, '');
            console.log(path);
            pathFiles.push(path.trim());
        }
    }
    const DEST_DIR = "./ALL";
    const DEST_DIR_LIBS = "./ALL/lib";
    if (!fs.existsSync(DEST_DIR)) {
        fs.mkdirSync(DEST_DIR);
    }
    if (!fs.existsSync(DEST_DIR_LIBS)) {
        fs.mkdirSync(DEST_DIR_LIBS);
    }
    // Copy files into folder
    for (let i = 0; i < pathFiles.length; i++) {
        await copyFile(pathFiles[i], DEST_DIR_LIBS);
    }
    await copyFile(executable, DEST_DIR);
    // Zip folder
    const zipRes = await runExecutable("zip", ["-r", output_name, DEST_DIR]);
}

analize();



