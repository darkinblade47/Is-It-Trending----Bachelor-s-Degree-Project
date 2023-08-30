import React from 'react';
import SpecificationComponent from './SpecificationComponent';
import RelevanceScore from '../RelevanceScore';

function Specifications(props) {
  const specs = props.specData;
  const url = 'https://corsproxy.io/?' + encodeURIComponent(specs.image);

  return (
    <>
      <div className='grid grid-cols-3 grid-flow-col gap-4 pb-3 max-lg:flex max-lg:flex-col'>
        <div className='col-span-2 bg-white shadow-lg rounded-sm border border-slate-200'>
          <div className='flex justify-center px-3'>
            <img src={url} alt="Laptop Image" height={500} width={500}/>
          </div>
          <p className="text-center mx-3">
            {specs.laptop_model}
          </p>
          <p className="text-center pb-3">
            <strong>Preț</strong>: {specs.price}
          </p>
        </div>

        <div className='col-span-1 max-lg:flex'>
          {<RelevanceScore score={specs.relevance_score} />}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4 justify-self-stretch pb-6 max-lg:flex max-lg:flex-col">

        {specs && Object.entries(specs.specification_data).map(([key, value]) => (
          <div key={key} className='p-3 bg-white shadow-lg rounded-sm border border-slate-200'>
            <SpecificationComponent data={JSON.stringify(value)} title={key} />
          </div>
        ))}
      </div>
    </>
  );
}

export default Specifications; 