echo "make_dataset.sh: \"Combining datasets and turning it into 100 files for easier KITRO refinement\""
python dataset.py
echo "make_dataset.sh: \"Running KITRO refinement (this will take a while... hours)\""
./run_refine
echo "make_dataset.sh: \"Congrats you are done with KITRO refinement!\"" 
cp data/refined000.pt data/merged000.pt
cp data/refined001.pt data/merged001.pt
cp data/refined002.pt data/merged002.pt
cp data/refined003.pt data/merged003.pt
cp data/refined004.pt data/merged004.pt
echo "make_dataset.sh \"merging the refined data into 5 files\""
./merge.sh
echo "make_dataset.sh" \"Complete!\""