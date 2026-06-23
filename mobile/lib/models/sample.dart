enum SampleStatus { pending, processing, completed }

class Sample {
  int? id;
  String idTag;
  String imagePath;
  double parasitemiaPercent;
  SampleStatus status;
  int totalCells;
  int infectedCells;

  Sample({
    this.id,
    required this.idTag,
    required this.imagePath,
    this.parasitemiaPercent = 0.0,
    this.status = SampleStatus.pending,
    this.totalCells = 0,
    this.infectedCells = 0,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'idTag': idTag,
      'imagePath': imagePath,
      'parasitemiaPercent': parasitemiaPercent,
      'status': status.toString().split('.').last,
      'totalCells': totalCells,
      'infectedCells': infectedCells,
    };
  }

  factory Sample.fromMap(Map<String, dynamic> map) {
    return Sample(
      id: map['id'] as int?,
      idTag: map['idTag'] as String,
      imagePath: map['imagePath'] as String,
      parasitemiaPercent: (map['parasitemiaPercent'] as num?)?.toDouble() ?? 0.0,
      status: _statusFromString(map['status'] as String? ?? 'pending'),
      totalCells: (map['totalCells'] as int?) ?? 0,
      infectedCells: (map['infectedCells'] as int?) ?? 0,
    );
  }

  static SampleStatus _statusFromString(String s) {
    switch (s) {
      case 'processing':
        return SampleStatus.processing;
      case 'completed':
        return SampleStatus.completed;
      default:
        return SampleStatus.pending;
    }
  }
}
