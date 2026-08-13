CREATE PROCEDURE [dbo].[HRMS_GetAllPositionByTeamId.1.0.0]
	@teamId INT
AS
BEGIN 
	SET NOCOUNT ON
    SELECT [Id], [PositionName], [TeamId]
    FROM [dbo].[Position]
    WHERE [TeamId] = @teamId
END