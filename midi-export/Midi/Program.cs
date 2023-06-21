#define READ_MODE 

using Melanchall.DryWetMidi.Common;
using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;
using Melanchall.DryWetMidi.MusicTheory;
using Note = Melanchall.DryWetMidi.Interaction.Note;


#if !READ_MODE
const double timeBias = 350;
#else
const double timeBias = 0.25;
#endif
var beats = new List<MidiBeat>();
var transformChannel = new Dictionary<int, int>();
using (var sr = new StreamReader(args.Last())) {
    while (!sr.EndOfStream) {
        var ipt = sr.ReadLine()!;
        if (string.IsNullOrWhiteSpace(ipt)) continue;
        try {
            var s = ipt.Split(",").Take(4).Select(double.Parse).ToList();
            #if !READ_MODE
            var b = new MidiBeat {
                Level = (int)s[3],
                Instrument = (int)s[2],
                StartTime = (int)(s[0] / timeBias),
                RemainTime = (int)((s[1] - s[0]) / timeBias)
            };
            #else
            var b = new MidiBeat {
                Level = (int)s[0],
                Instrument = (int)s[1],
                StartTime = (int)(s[2] / timeBias),
                RemainTime = (int)(s[3] / timeBias)
            };
            #endif
            beats.Add(b);
            transformChannel.TryAdd(b.Instrument, transformChannel.Count);
        }
        catch (FormatException e) {
            Console.WriteLine(ipt);
             /* Ignore */
        }
    }
}

var midi = new MidiFile();
var tempoMap = midi.GetTempoMap();

foreach (var k in transformChannel.Keys) {
    Console.WriteLine($"Processing instrument {k}");
    var chunk = new TrackChunk();
    midi.Chunks.Add(chunk);
    using var notesManager = chunk.ManageNotes();
    var notes = notesManager.Objects;
    foreach(var x in beats.Where(x => x.Instrument == k))
        notes.Add(new Note((SevenBitNumber)x.Level, x.RemainTime, x.StartTime) {
            
        });
}

var outputFile = Path.GetFileNameWithoutExtension(args.Last()) + ".mid";
File.Delete(outputFile);
midi.Write(outputFile);

internal struct MidiBeat {
    public int Level;
    public int Instrument;
    public int StartTime;
    public int RemainTime;
}