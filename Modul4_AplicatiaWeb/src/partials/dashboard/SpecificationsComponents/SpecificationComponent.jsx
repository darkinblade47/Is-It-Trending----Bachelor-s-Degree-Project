import React from 'react';

function SpecificationComponent(props){

    const specData = props.data;
    const title = props.title;

    return (
        <>
          <header className="p-3 border-b border-slate-100 flex items-center">
                <h2 className="font-bold text-slate-800">{title}</h2>
              </header>
          {specData && Object.entries(JSON.parse(specData)).map(([key, value]) => (
            
            <div key={key}>
            <p className='ml-3'>
              <strong>{key}</strong> : {value} 
            </p>
          </div>
        ))}
      </>
      );
}

export default SpecificationComponent;
