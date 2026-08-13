CREATE PROCEDURE [dbo].[HRMS_GetAllDepartment.1.0.0]
AS
BEGIN 
	SET NOCOUNT ON
    SELECT 
        d.[Id], 
        d.[DepartmentName], 
        Count(t.TeamId) as TotalTeam
    FROM [dbo].[Department] d
    LEFT JOIN Team t ON d.Id = t.DepartmentId
    GROUP BY  
    	d.[Id], 
        d.[DepartmentName]
END