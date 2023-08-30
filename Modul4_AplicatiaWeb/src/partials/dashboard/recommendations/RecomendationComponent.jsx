import React from 'react';

function RecommendationComponent(props){
    const data = props.data;
    const url = 'https://corsproxy.io/?' + encodeURIComponent(data.image);

    return (
        <>
        {url &&

            <div className='flex flex-col col-span-1 m-3'>
                <div className='flex justify-center'>
                <img src={url} alt="Laptop Image" width={240} height={240}/>
                </div>
                    <p className="text-center">
                    {data.laptop_model}    
                    </p>
                    <p className="text-center">
                        <strong>Indice de relevanță</strong>: {Math.ceil(data.relevance_score*100)/100}
                    </p>
                    <p className="text-center">
                    <strong>Preț</strong>: {data.price}
                    </p>
            </div>
            }
        </>

    );
}

export default RecommendationComponent;