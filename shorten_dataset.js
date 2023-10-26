const { readFileSync, writeFileSync } = require("fs");

const MAX_LENGTH = 3000;

const p = "prepared/prepared_1697562094237.json";
const out = "prepared/prepared_1697562094237-short.json";

const json = JSON.parse(readFileSync(p, "utf-8"));

const outJson = json.slice(0, MAX_LENGTH);

writeFileSync(out, JSON.stringify(outJson));
