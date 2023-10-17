const { GenbankSequencesFile } = require("biotech-js");
const { readdir, writeFile } = require("fs/promises");

DIR_TO_SAVE = "prepared";

class Record {
	constructor(className, sequence) {
		this.className = className;
		this.sequence = sequence;
	}

	toJSON() {
		return {
			className: this.className,
			sequence: this.sequence,
		};
	}
}

const parse = async (path) => {
	const file = new GenbankSequencesFile(path);
	await file.process();

	const dict = {};
	console.log(
		`took ms: ${file.tookMs}, processing status: ${file.processingStatus}, loaded ${file.sequencesNumber} sequences`
	);
	const filteredOut = file.sequences.filter((s) => s.Source !== undefined);

	let sequences = filteredOut.filter(
		(s) => s.Source.Organism[0] === "Bacteria"
	);
	sequences.forEach((sequence) => {
		const organisms = sequence.Source.Organism;
		const desiredOrg = organisms[1];
		if (dict[desiredOrg]) {
			dict[desiredOrg] += 1;
		} else {
			dict[desiredOrg] = 1;
		}
		sequence["className"] = desiredOrg;
	});

	const familyNames = Object.keys(dict).filter((name) => {
		return name !== "undefined" && dict[name] > 30;
	});

	sequences = sequences.filter((sequence) =>
		familyNames.includes(sequence["className"])
	);
	console.log(sequences.length);

	return sequences.map((s) => {
		return new Record(s["className"], s.Origin);
	});
};

(async () => {
	const files = await readdir("./databases");
	const filtered = files.filter((f) => f.endsWith(".gb"));
	const records = [];
	for (const file of filtered) {
		const p = `./databases/${file}`;
		console.log(p);
		const s = await parse(p);
		records.push(...s);
	}
	const filename = `prepared_${new Date().getTime()}.json`;
	await writeFile(`${DIR_TO_SAVE}/${filename}`, JSON.stringify(records));
})();
