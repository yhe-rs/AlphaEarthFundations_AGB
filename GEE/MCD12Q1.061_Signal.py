import os
from tqdm import tqdm
from datetime import datetime, timedelta
import ee

ee.Authenticate()
ee.Initialize(project='fates-mrv')


if __name__ == "__main__":
    
    # define chunk size
    CHUNK_SIZE = 1 # per cell per chunk
    
    # gee assets path
    parent ='projects/fates-mrv/assets'
    
    # Ask GEE for the list of *direct* children of that folder
    resp = ee.data.listAssets({'parent': parent})  # dict with key 'assets'
    assets_meta = resp.get('assets', [])           # list of dictionaries
    
    # Keep only assets whose type == 'TABLE'   (vectors/shapefiles)
    table_ids = [a['name'] for a in assets_meta if a['type'] == 'TABLE']
    print('Found', len(table_ids), 'vector assets')
    
    # Print all found asset IDs with index
    for i, aid in enumerate(table_ids):
        print(f'{i:2d}. {aid}')
    
    year = 2017    
    START_DATE = f'{year}-01-01'
    END_DATE   = f'{year}-12-31'

    
    #  Read them in a loop
    for asset_id in tqdm(table_ids):
        #print(table_ids)

        # Filter to process only assets
        if not asset_id.endswith('INFYS_Arbolado_2015_2020_geo_2018'): # get table endwith fishnet
            continue  # Skip if it does not match this pattern

        # Extract basename from asset ID for naming outputs
        basename = os.path.basename(asset_id)      # ← fishnet
        print('\n───────────────────────────────────────────────')
        print('Processing:', asset_id)

        # Load the vector asset as a FeatureCollection
        fishnet = ee.FeatureCollection(asset_id)
        # Get number of features (grid cells) in the collection (client-side)
        total_cells = fishnet.size().getInfo()
        print('\nNumber of grid cells :', total_cells)
        print('Schema columns:', fishnet.first().propertyNames().getInfo())

        # spatial tiling : loop over the grid in chunks
        for offset in range(0, total_cells, CHUNK_SIZE):
            chunk_idx = offset // CHUNK_SIZE
            chunk_fc  = ee.FeatureCollection(fishnet.toList(CHUNK_SIZE, offset))
            
            # Extract geometry of chunk_fc
            chunk_geom = chunk_fc.geometry()

            # Filter the ImageCollection to images intersecting the polygon geometry
            modis_collection = (ee.ImageCollection("MODIS/061/MCD12Q1")
                               .filterBounds(chunk_geom)  # geom touched scene
                                .filterDate(START_DATE, END_DATE)
                               # .select(['LC_Type1','LC_Type2'])
                               # .map(lambda image: image.toFloat())
                               # .map(lambda image: image.toDouble())
                              )
            print('Bands info:', modis_collection.first().bandNames().getInfo())
            # print('Bands info:', modis_collection.first().getInfo())


            # Check bands before reduction: get the number of images in the filtered collection (client-side)
            collection_size = modis_collection.size().getInfo()
            if collection_size == 0:
                 # No images found for this polygon, skip processing
                print(f"No images found for chunk {chunk_idx}. Skipping...")
                continue  # skip to next chunk

            else:
                print(f"⚠️  modis collection_size: {collection_size}")
                # Map over the ImageCollection to perform reduceRegions on each image
                # and add a 'date' property to each resulting feature
                def addDateAndReduce(image):
                    # Perform reduceRegions on the current image
                    reduced_features = image.reduceRegions(
                        collection=chunk_fc,  # The chunk of features
                        reducer=ee.Reducer.median(),  # median per feature (polygon)
                        scale=10,  # Resolution,
                        crs='EPSG:3857',
                        tileScale=1
                    )
                    # Add the date as a property to each feature
                    return reduced_features.map(lambda feature: feature.set('date', image.date().format('YYYY-MM-dd')))
                
                # Apply the function to each image in the collection and flatten the result
                modis_samples = (modis_collection.map(lambda image: image)  # Fix: Map clip to each image
                                                .map(addDateAndReduce)
                                                .flatten()
                                )
                print('⚠️  MODIS samples size:', modis_samples.size().getInfo())

                # # Print the features in MODIS samples
                # features = modis_samples.toList(modis_samples.size())
                # feature_count = features.size().getInfo()
                # for i in range(feature_count):
                #     feature = ee.Feature(features.get(i))
                #     print(feature.getInfo())

                
                # Create a descriptive task name for export
                task_desc = f'modis_{basename}_chunk_{chunk_idx}'
    
                # Set up and start an export task to Google Drive
                task = ee.batch.Export.table.toDrive(**{        
                    'collection': modis_samples,
                    'description': task_desc, # set file name
                    'folder': 'modis',  # folder on google drive
                    'fileFormat': 'csv'  # format
                })
    
                task.start()
                
                print(f'\n    → task: {task_desc} submitted '
                      f'({offset:,} – {min(offset+CHUNK_SIZE, total_cells)-1:,})')
                    
                # Optional: sleep to avoid submitting too many tasks too quickly
                # Each project's queue supports a maximum of 3,000 tasks
                import time
                time.sleep(3) # sleep interval to aviod creating multi outdir in google drive
                # do a pre-check of time/chunk to set this sleep value
