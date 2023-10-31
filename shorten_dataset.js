const { readFileSync, writeFileSync } = require("fs");

// max is the min of the number of items per class, so around 34
const ITEMS_PER_CLASS = 20;

const p = "prepared/prepared_1697562094237.json";
const out = "prepared/prepared_1697562094237-short.json";

const json = JSON.parse(readFileSync(p, "utf-8"));

let dict = {};
const outJson = json
	.map((item) => {
		const { c } = item;
		if (dict[c] === undefined) {
			dict[c] = 0;
		}
		if (dict[c] < ITEMS_PER_CLASS) {
			dict[c]++;
			return item;
		}
		return null;
	})
	.filter((item) => item !== null);
writeFileSync(out, JSON.stringify(outJson));
