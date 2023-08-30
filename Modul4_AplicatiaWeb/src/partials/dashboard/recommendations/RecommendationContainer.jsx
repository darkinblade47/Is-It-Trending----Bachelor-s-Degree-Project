import { React, useEffect, useState, useRef } from "react";
import Info from "../../../utils/Info";
import RecommendationComponent from "./RecomendationComponent";

function RecommendationContainer(props) {
    let minPrice = useRef(null);
    let maxPrice = useRef(null);
    const [recData, setRecData] = useState(null);
    let service = props.service;
    let [status, setStatus] = useState("");
    let [error, setError] = useState("");

    const handleGetRecommendations = () => {
        if (minPrice.current !== null && maxPrice.current !== null) {
            if(parseInt(minPrice.current) === parseInt(maxPrice.current)) {
                setError("Prețurile nu pot fi identice!")
            }
            else if (parseInt(maxPrice.current) < parseInt(minPrice.current))
            {
                setError("Prețul maxim trebuie să fie mai mare decât cel minim!");
                
            }
            else if (parseInt(maxPrice.current) < 0 || parseInt(minPrice.current) < 0){
                setError("Prețurile nu pot avea valori negative!");
            }
            else{
                setError("");
                service.getRecommendation(parseInt(minPrice.current), parseInt(maxPrice.current)).then((response) => { setRecData(response); });
            }
        }
        else {
            setError("Introduceți intervalul de preț!");
        }

    }

    const handleMinPrice = (event) => {
        minPrice.current = event.target.value;
    }

    const handleMaxPrice = (event) => {
        maxPrice.current = event.target.value;
    }

    useEffect(() => {
        if (recData !== null)
        {
            if (recData.recommended.length >0 || recData.similar.length > 0)
                setStatus("")
            else
                setStatus("Nu avem recomandări în intervalul de preț selectat sau pentru produsul de interes.")
        }
        else
        {
            setStatus("")
        }

    }, [recData, service])

    return (
        <>
            <div className="flex flex-col col-span-full bg-white shadow-lg rounded-sm border border-slate-200">
                <header className="px-5 py-4 border-b border-slate-100 flex items-center">
                    <h2 className="font-semibold text-slate-800">Produse recomandate și similare</h2>
                    <Info className="ml-2" containerClassName="min-w-44">
                        <div className="text-sm">Produsele indicate sunt alese după performanță și indicele lor de relevanță.</div>
                    </Info>
                </header>

                <div className="flex flex-row row-span-4 gap-3 justify-evenly px-5">
                    <div className="flex flex-col col-span-4 gap-6 justify-start items-center border-r pr-5 pb-5">
                        <label className="text-center font-medium pt-2">Interval de preț:</label>
                        <input type="text" id="first_name" className="
                        bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 
                        focus:border-blue-500 block w-36 p-2.5" placeholder="Dați prețul minim:" required onChange={handleMinPrice}/>
                        <input type="text" id="first_name" className="
                        bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 
                        focus:border-blue-500 block w-36 p-2.5" placeholder="Dați prețul maxim:" required onChange={handleMaxPrice}/>

                        <button className="bg-white border border-gray-300 text-gray-900 text-sm rounded-lg 
                        focus:ring-blue-500focus:border-blue-500 block w-36 p-2.5" onClick={handleGetRecommendations} >Caută recomandări</button>
                        {error && <p className="text-center text-red-600 font-semibold">{error}</p>}
                    </div>
                    <div className="w-full">
                        {status && <p className="flex h-full text-center items-center justify-center ">{status}</p>}
                        {recData && (recData.recommended.length > 0 || recData.similar.length > 0) && Object.keys(recData).map((key) => (
                            <>
                            {key ==="recommended" && recData.recommended.length > 0 && <h2 className="font-semibold text-xl text-slate-800">Produse recomandate</h2>}
                            {key ==="similar" && recData.similar.length > 0 && <h2 className="font-semibold text-xl text-slate-800">Produse similare</h2>}
                            {/* <h2 className="font-semibold text-xl text-slate-800">{key ==="recommended" ? "Produse recomandate" : "Produse similare"}</h2> */}
                            <div className="flex flex-row flex-wrap pb-9 max-lg:flex-col">
                                {recData[key].length > 0 && recData[key].map((recommendation, index)=> (
                                    <div className="flex w-1/3 p-2 max-lg:w-full">
                                        <div className="rounded-sm border border-slate-200 shadow-lg overflow-hidden cursor-pointer hover:scale-105 hover:border-indigo-500 transition-transform duration-300" onClick={() => (props.linkHandle(recommendation.product_url))}>
                                            <RecommendationComponent data={recommendation} />
                                        </div>
                                    </div>
                                ))}
                            </div></>
                        ))}
                    </div>
                </div>
            </div>
        </>
    );
}

export default RecommendationContainer;