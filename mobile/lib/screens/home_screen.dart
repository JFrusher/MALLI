import 'package:flutter/material.dart';
import '../database/database_helper.dart';
import '../models/sample.dart';
import 'capture_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late Future<List<Sample>> _completedSamples;

  @override
  void initState() {
    super.initState();
    _loadCompleted();
  }

  void _loadCompleted() {
    _completedSamples = DatabaseHelper.instance.fetchCompletedSamples();
  }

  Future<void> _refresh() async {
    setState(() {
      _loadCompleted();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('MALLI — Completed Samples')),
      body: FutureBuilder<List<Sample>>(
        future: _completedSamples,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return Center(child: Text('No completed samples.'));
          }
          final items = snapshot.data!;
          return RefreshIndicator(
            onRefresh: _refresh,
            child: ListView.builder(
              itemCount: items.length,
              itemBuilder: (context, idx) {
                final s = items[idx];
                return ListTile(
                  title: Text(s.idTag),
                  subtitle: Text('${s.parasitemiaPercent.toStringAsFixed(1)}% RBCs Parasitized'),
                );
              },
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          await Navigator.push(context, MaterialPageRoute(builder: (_) => const CaptureScreen()));
          _refresh();
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}
