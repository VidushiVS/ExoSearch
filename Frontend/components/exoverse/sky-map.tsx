'use client';
import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Exoplanet } from '@/lib/types';

type SkyMapProps = {
  planets: Exoplanet[];
  onSelectPlanet: (planet: Exoplanet) => void;
  selectedPlanet: Exoplanet | null;
};

const predictionColors: Record<Exoplanet['prediction'], THREE.Color> = {
  'Confirmed': new THREE.Color(0x00FFFF), // Cyan
  'Candidate': new THREE.Color(0xFFa500), // Orange
  'False Positive': new THREE.Color(0xFF4444), // Red
};
const highlightColor = new THREE.Color('#FFFF00'); // Yellow

export function SkyMap({ planets, onSelectPlanet, selectedPlanet }: SkyMapProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const planetsGroupRef = useRef<THREE.Group | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const starFieldsRef = useRef<THREE.Points[]>([]);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const lastHoveredPlanetIdRef = useRef<string | null>(null);
  const scaleRef = useRef<HTMLDivElement | null>(null);

  const [textures, setTextures] = useState<{ [key: string]: THREE.CanvasTexture | null }>({});
  
  const createGlowTexture = (color: string) => {
      const canvas = document.createElement('canvas');
      canvas.width = 128;
      canvas.height = 128;
      const context = canvas.getContext('2d');
      if (!context) return null;
  
      const gradient = context.createRadialGradient(64, 64, 0, 64, 64, 64);
      gradient.addColorStop(0.1, color);
      gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.1)');
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      
      context.fillStyle = gradient;
      context.fillRect(0, 0, 128, 128);
  
      return new THREE.CanvasTexture(canvas);
  }

  useEffect(() => {
    // This code runs only on the client
    if (typeof window !== 'undefined') {
        setTextures({
          'Confirmed': createGlowTexture('rgba(0, 255, 255, 0.6)'),
          'Candidate': createGlowTexture('rgba(255, 165, 0, 0.6)'),
          'False Positive': createGlowTexture('rgba(255, 68, 68, 0.6)'),
        });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!mountRef.current) return;
    const currentMount = mountRef.current;
    
    // === Scene Setup ===
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, currentMount.clientWidth / currentMount.clientHeight, 0.1, 4000);
    camera.position.z = 1000;
    cameraRef.current = camera;
    
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    currentMount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const planetsGroup = new THREE.Group();
    scene.add(planetsGroup);
    planetsGroupRef.current = planetsGroup;

    // === Controls ===
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.04;
    controls.enablePan = false;
    controls.enableZoom = true;
    controls.minDistance = 0.1;
    controls.maxDistance = 4000;
    controls.rotateSpeed = 0.5;
    controls.zoomSpeed = 1.2;
    controlsRef.current = controls;

    // === Starfield ===
    const createStarField = (count: number, size: number, distance: number) => {
        const starsGeometry = new THREE.BufferGeometry();
        const starsVertices = [];
        for (let i = 0; i < count; i++) {
          const x = (Math.random() - 0.5) * 2;
          const y = (Math.random() - 0.5) * 2;
          const z = (Math.random() - 0.5) * 2;
          const vec = new THREE.Vector3(x, y, z).normalize().multiplyScalar(distance);
          starsVertices.push(vec.x, vec.y, vec.z);
        }
        starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
        const starsMaterial = new THREE.PointsMaterial({ color: 0xAAAAAA, size: size, transparent: true, opacity: 0.8 });
        return new THREE.Points(starsGeometry, starsMaterial);
    };
    starFieldsRef.current = [
      createStarField(5000, 0.1, 1500),
      createStarField(5000, 0.2, 2000),
      createStarField(2000, 0.3, 2500),
    ];
    starFieldsRef.current.forEach(sf => scene.add(sf));
    

    // === Tooltip ===
    const tooltip = document.createElement('div');
    tooltip.className = "absolute text-white p-2 rounded-md bg-black/50 text-xs pointer-events-none hidden font-mono backdrop-blur-sm border border-primary/20";
    currentMount.appendChild(tooltip);
    tooltipRef.current = tooltip;

    const scaleIndicator = document.createElement('div');
    scaleIndicator.className = "absolute bottom-12 left-4 text-xs text-muted-foreground bg-black/30 p-2 rounded-md font-mono";
    currentMount.appendChild(scaleIndicator);
    scaleRef.current = scaleIndicator;

    // === Animation Loop ===
    const animate = () => {
      animationFrameId.current = requestAnimationFrame(animate);
      controls.update();
      starFieldsRef.current[0].rotation.y += 0.00002;
      starFieldsRef.current[1].rotation.y += 0.00003;
      starFieldsRef.current[2].rotation.y += 0.00004;

      if (scaleRef.current && cameraRef.current) {
        const distance = cameraRef.current.position.length();
        const scaleValue = (distance / 3000) * 50; // Arbitrary scaling for display
        scaleRef.current.textContent = `Scale: ~${scaleValue.toFixed(2)} AU`;
      }


      // Smoothly update planet visuals
      if (planetsGroupRef.current) {
        planetsGroupRef.current.children.forEach(child => {
          const container = child as THREE.Group;
          const sphere = container.children.find(c => c.name === 'planet_body') as THREE.Mesh;
          const glow = container.children.find(c => c.name === 'planet_glow') as THREE.Sprite;

          if (sphere && glow && sphere instanceof THREE.Mesh) {
              const isSelected = container.userData.planetId === selectedPlanet?.id;
              const isHovered = container.userData.planetId === lastHoveredPlanetIdRef.current;

              const targetScale = isSelected ? 1.5 : (isHovered ? 1.3 : 1);
              const planet = planets.find(p => p.id === container.userData.planetId);
              const baseColor = planet ? (predictionColors[planet.prediction] || new THREE.Color(0x999999)) : new THREE.Color(0x999999);
              const targetColor = isSelected ? highlightColor : baseColor;
              const targetGlowOpacity = isSelected ? 0.9 : (isHovered ? 0.8 : 0.5);

              sphere.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
              (sphere.material as THREE.MeshBasicMaterial).color.lerp(targetColor, 0.1);
              glow.material.opacity = THREE.MathUtils.lerp(glow.material.opacity, targetGlowOpacity, 0.1);
          }
        });
      }

      renderer.render(scene, camera);
    };
    animate();

    // === Event Listeners ===
    const handleResize = () => {
      if (!currentMount || !rendererRef.current || !cameraRef.current) return;
      cameraRef.current.aspect = currentMount.clientWidth / currentMount.clientHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(currentMount.clientWidth, currentMount.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    const raycaster = new THREE.Raycaster();
    const handleMouseMove = (event: MouseEvent) => {
        if (!currentMount || !cameraRef.current || !planetsGroupRef.current || !tooltipRef.current) return;
        const mouse = new THREE.Vector2();
        const rect = currentMount.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / currentMount.clientWidth) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / currentMount.clientHeight) * 2 + 1;
        
        raycaster.setFromCamera(mouse, cameraRef.current);
        const planetMeshes = planetsGroupRef.current.children.map(c => c.children[0]); // Target the sphere
        const intersects = raycaster.intersectObjects(planetMeshes);

        if (intersects.length > 0) {
            const planetId = intersects[0].object.userData.planetId;
            const planet = planets.find(p => p.id === planetId);
            lastHoveredPlanetIdRef.current = planetId;

            if (planet && tooltipRef.current) {
                tooltipRef.current.style.display = 'block';
                tooltipRef.current.style.left = `${event.clientX - rect.left + 15}px`;
                tooltipRef.current.style.top = `${event.clientY - rect.top + 15}px`;
                tooltipRef.current.innerHTML = `
                    <strong class="text-primary">${planet.id}</strong><br/>
                    Status: ${planet.prediction}<br/>
                    <em class="text-primary/70">Click to explore</em>
                `;
            }
        } else {
            if (tooltipRef.current) tooltipRef.current.style.display = 'none';
            lastHoveredPlanetIdRef.current = null;
        }
    };
    
    const handleClick = (event: MouseEvent) => {
        if (!currentMount || !cameraRef.current || !planetsGroupRef.current) return;
        const mouse = new THREE.Vector2();
        const rect = currentMount.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / currentMount.clientWidth) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / currentMount.clientHeight) * 2 + 1;
        
        raycaster.setFromCamera(mouse, cameraRef.current);
        const planetMeshes = planetsGroupRef.current.children.map(c => c.children[0]); // Target the sphere
        const intersects = raycaster.intersectObjects(planetMeshes);

        if (intersects.length > 0) {
            const planetId = intersects[0].object.userData.planetId;
            const planet = planets.find(p => p.id === planetId);
            if (planet) onSelectPlanet(planet);
        }
    };
    
    currentMount.addEventListener('mousemove', handleMouseMove);
    currentMount.addEventListener('click', handleClick);
    
    // === Cleanup ===
    return () => {
      if(animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
      window.removeEventListener('resize', handleResize);
      currentMount.removeEventListener('mousemove', handleMouseMove);
      currentMount.removeEventListener('click', handleClick);
      controls.dispose();
      if (rendererRef.current && rendererRef.current.domElement.parentElement === currentMount) {
        currentMount.removeChild(rendererRef.current.domElement);
      }
      if (tooltipRef.current && tooltipRef.current.parentElement === currentMount) {
        currentMount.removeChild(tooltipRef.current);
      }
      if (scaleRef.current && scaleRef.current.parentElement === currentMount) {
        currentMount.removeChild(scaleRef.current);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update planets when data changes
  useEffect(() => {
    if (!planetsGroupRef.current || Object.keys(textures).length === 0) return;
    const group = planetsGroupRef.current;
    
    while(group.children.length > 0){ 
      const child = group.children[0];
      // Proper disposal of materials and geometries
      if (child instanceof THREE.Group) {
          child.children.forEach((mesh: any) => {
              mesh.geometry?.dispose();
              if (Array.isArray(mesh.material)) {
                mesh.material.forEach((m: any) => m.dispose());
              } else if (mesh.material){
                mesh.material?.dispose();
              }
          });
      }
      group.remove(child);
    }

    planets.forEach(planet => {
      const container = new THREE.Group();
      
      const planetColor = predictionColors[planet.prediction] || new THREE.Color(0x999999);
      const glowTexture = textures[planet.prediction] || textures['Candidate'];

      const planetRadius = Math.max(0.2, planet.planetRadius * 0.2);
      const geometry = new THREE.SphereGeometry(planetRadius, 16, 16);
      const material = new THREE.MeshBasicMaterial({ color: planetColor });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.name = 'planet_body';
      sphere.userData = { planetId: planet.id };
      
      container.add(sphere);

      if (glowTexture) {
        const glowMaterial = new THREE.SpriteMaterial({
          map: glowTexture,
          color: planetColor,
          blending: THREE.AdditiveBlending,
          opacity: 0.5,
        });
        const glow = new THREE.Sprite(glowMaterial);
        const glowSize = planetRadius * 6;
        glow.scale.set(glowSize, glowSize, glowSize);
        glow.name = 'planet_glow';
        container.add(glow);
      }

      const raRad = planet.ra * (Math.PI / 180);
      const decRad = planet.dec * (Math.PI / 180);
      
      // Plot false positives far away
      const isFalsePositive = planet.distanceFromStar < 0;
      const displayDistance = isFalsePositive ? 1400 + Math.random() * 100 : 50 + planet.distanceFromStar * 150;

      container.position.x = displayDistance * Math.cos(decRad) * Math.cos(raRad);
      container.position.z = displayDistance * Math.cos(decRad) * Math.sin(raRad);
      container.position.y = displayDistance * Math.sin(decRad);
      
      container.userData = { planetId: planet.id };
      group.add(container);
    });
  }, [planets, textures]);

  useEffect(() => {
    if (selectedPlanet) {
        lastHoveredPlanetIdRef.current = selectedPlanet.id;
    }
  }, [selectedPlanet])

  return (
    <div className='w-full h-full relative'>
      <div ref={mountRef} className="h-full w-full cursor-crosshair active:cursor-grabbing" />
      <div className="absolute top-4 right-4 flex gap-4">
        <div className="glassmorphism rounded-lg px-4 py-2 text-center">
            <p className="text-2xl font-bold font-mono text-primary">{planets.filter(p => p.prediction !== 'False Positive').length}</p>
            <p className="text-muted-foreground text-sm">Visible Objects</p>
        </div>
      </div>
      <div className="absolute bottom-4 left-4 text-xs text-muted-foreground bg-black/30 p-2 rounded-md font-mono">
        Left-click + drag to rotate. Scroll to zoom.
      </div>
    </div>
  );
}
