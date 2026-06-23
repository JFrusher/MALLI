import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';
import 'package:path_provider/path_provider.dart';
import '../models/sample.dart';

class DatabaseHelper {
  static final DatabaseHelper instance = DatabaseHelper._internal();
  factory DatabaseHelper() => instance;
  DatabaseHelper._internal();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final docs = await getApplicationDocumentsDirectory();
    final path = p.join(docs.path, 'malli_samples.db');
    return await openDatabase(
      path,
      version: 2,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        idTag TEXT NOT NULL,
        imagePath TEXT NOT NULL,
        parasitemiaPercent REAL DEFAULT 0.0,
        status TEXT NOT NULL,
        totalCells INTEGER DEFAULT 0,
        infectedCells INTEGER DEFAULT 0
      )
    ''');
  }

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    if (oldVersion < 2) {
      await db.execute('ALTER TABLE samples ADD COLUMN totalCells INTEGER DEFAULT 0');
      await db.execute('ALTER TABLE samples ADD COLUMN infectedCells INTEGER DEFAULT 0');
    }
  }

  Future<int> insertSample(Sample sample) async {
    final db = await database;
    return await db.insert('samples', sample.toMap());
  }

  Future<List<Sample>> fetchCompletedSamples() async {
    final db = await database;
    final maps = await db.query('samples', where: 'status = ?', whereArgs: ['completed'], orderBy: 'id DESC');
    return maps.map((m) => Sample.fromMap(m)).toList();
  }

  Future<int> updateSample(Sample sample) async {
    final db = await database;
    return await db.update('samples', sample.toMap(), where: 'id = ?', whereArgs: [sample.id]);
  }

  /// Updates the entry to completed with analysis results, then deletes the image file.
  Future<void> markCompletedAndDeleteImage(
    int id,
    double percent,
    String imagePath, {
    int totalCells = 0,
    int infectedCells = 0,
  }) async {
    final db = await database;
    await db.update('samples', {
      'parasitemiaPercent': percent,
      'status': 'completed',
      'totalCells': totalCells,
      'infectedCells': infectedCells,
    }, where: 'id = ?', whereArgs: [id]);

    try {
      final f = File(imagePath);
      if (await f.exists()) {
        await f.delete();
      }
    } catch (e) {
      // swallow errors for now; caller may log if desired
    }
  }

  Future<void> deleteSampleById(int id) async {
    final db = await database;
    await db.delete('samples', where: 'id = ?', whereArgs: [id]);
  }
}
