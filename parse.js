const { readFileSync } = require("fs");
const { FastaSequenceFile } = require("biotech-js");

const parse = async (path) => {
	const file = new FastaSequenceFile(path);
	await file.process();
	const a = file.sequences;
	const sequencesChunked = a.map(({ description, sequence }) => {
		const newSequence = [];
		for (let i = 0; i < sequence.length; i += 5) {
			newSequence.push(sequence.substr(i, 5));
		}
		return {
			className: "",
			sequence: newSequence,
		};
	});
};

(async () => {
	await parse("databases/swissprot.fasta");
})();
