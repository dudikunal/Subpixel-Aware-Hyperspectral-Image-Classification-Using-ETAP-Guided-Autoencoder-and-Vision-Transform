from scipy.io import loadmat
import argparse

def check_principal_components(dataset_name):
   
    dataset_files = {
        'Indian': './data/indian_pines_TAP.mat',
        'Salinas': './data/salinasTAP15PC.mat',
        'Pavia': './data/Pavia_30.mat'
    }
    
    
    if dataset_name not in dataset_files:
        raise ValueError("Unsupported dataset. Choose from ['Indian', 'Salinas', 'Pavia'].")
    
    # Load the .mat file
    data = loadmat(dataset_files[dataset_name])
    
    input_data = data['input']

    num_pcs = input_data.shape[2]

    print(f"Dataset: {dataset_name}")
    print(f"Number of Principal Components (PCs): {num_pcs}")#gives no. of PC's
    print(f"Input data shape: {input_data.shape} (Height, Width, Bands)")

def main():
  
    parser = argparse.ArgumentParser(description="Check number of principal components in dataset")
    parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Salinas'], default='Pavia', help='dataset to use')
    args = parser.parse_args()
    
    check_principal_components(args.dataset)

if __name__ == '__main__':
    main()