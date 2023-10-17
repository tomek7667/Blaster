const { readFileSync, writeFileSync } = require("fs");

const p = "prepared/prepared_1697562094237.json";

const json = JSON.parse(readFileSync(p, "utf-8"));

const dict = {};
json.forEach((record) => {
	if (dict[record.className]) {
		dict[record.className] += 1;
	} else {
		dict[record.className] = 1;
	}
});

console.log(dict);
writeFileSync("dict.json", JSON.stringify(dict));
